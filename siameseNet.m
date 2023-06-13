%% EE 259 Pill Project CNN

clear; clearvars; close all; clc

%% Folder Information
% ===== WHAT YOU CHANGE ===== %
dataFolder = sprintf('pills'); % this is the big folder all the data goes in
imNum = 250; % number of images per pill
totalPills = 15; % total number of pills being classified
trainImg = 200; % number of images per pill used for training
testImg = 50; % number of images used per pill for testing
res = 224; % size of images
% =========================== %

%% CNN Set Up
imds_rgb = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'FileExtensions', {'.jpg'}, 'LabelSource', 'foldernames','ReadFcn', @readRGBImage);
imds_depth = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'FileExtensions', {'.jpg'}, 'LabelSource', 'foldernames', 'ReadFcn', @readDepthImage);
disp("hi")
% Get the list of RGB and depth image files
rgb_files = imds_rgb.Files;
depth_files = imds_depth.Files;

% Sort the file names
[~, rgb_indices] = sort(rgb_files);
[~, depth_indices] = sort(depth_files);

% Get the sorted file names
sorted_rgb_files = rgb_files(rgb_indices);
sorted_depth_files = depth_files(depth_indices);
disp("popo")
% Split the sorted RGB and depth image files
p = trainImg / imNum;
rgb_train_files = sorted_rgb_files(1:round(p*numel(sorted_rgb_files)));
depth_train_files = sorted_depth_files(1:round(p*numel(sorted_depth_files)));
rgb_test_files = sorted_rgb_files(round(p*numel(sorted_rgb_files))+1:end);
depth_test_files = sorted_depth_files(round(p*numel(sorted_depth_files))+1:end);
disp("jeje")
imdsTrain_rgb = imageDatastore(rgb_train_files, 'LabelSource', 'foldernames', 'ReadFcn', @readRGBImage);
imdsTrain_depth = imageDatastore(depth_train_files, 'LabelSource', 'foldernames', 'ReadFcn', @readDepthImage);
imdsTest_rgb = imageDatastore(rgb_test_files, 'LabelSource', 'foldernames', 'ReadFcn', @readRGBImage);
imdsTest_depth = imageDatastore(depth_test_files, 'LabelSource', 'foldernames', 'ReadFcn', @readDepthImage);

% Split the training datastore into training and validation
[imdsTrain_rgb, imdsVali_rgb] = splitEachLabel(imdsTrain_rgb, (9/10), 'randomized');
[imdsTrain_depth, imdsVali_depth] = splitEachLabel(imdsTrain_depth, (9/10), 'randomized');
imdsTrain = combine(imdsTrain_rgb, imdsTrain_depth);
imdsVali = combine(imdsVali_rgb, imdsVali_depth);
imdsTest = combine(imdsTest_rgb, imdsTest_depth);

% Get the number of observations in the datastores
numObservations = min(numel(imds_rgb.Files), numel(imds_depth.Files));
data = cell(numObservations, 3);
for i = 1:numObservations
   rgbImage = read(imds_rgb);
   depthImage = read(imds_depth);
   labels = imds_rgb.Labels(i);
   data{i, 1} = rgbImage;
   data{i, 2} = depthImage;
   data{i, 3} = labels;
end

% Create the custom datastore
customDS = CustomSiameseDatastore(data);

%% Define the Siamese network architecture
inputSize = [res res 3]; % Size of RGB images
inputDepthSize = [res res 1]; % Size of depth images

% RGB branch
rgbBranch = [
    imageInputLayer(inputSize, 'Name', 'input_rgb')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_rgb')
    reluLayer('Name', 'relu1_rgb')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1_rgb')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_rgb')
    reluLayer('Name', 'relu2_rgb')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2_rgb')
];

% Depth branch
depthBranch = [
    imageInputLayer(inputDepthSize, 'Name', 'input_depth')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_depth')
    reluLayer('Name', 'relu1_depth')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1_depth')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_depth')
    reluLayer('Name', 'relu2_depth')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2_depth')
];

% Concatenate the outputs of both branches
mergedLayer = concatenationLayer(3, 2, 'Name', 'merge');
finalLayers = [
    fullyConnectedLayer(totalPills, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Connect the branches and final layers
lgraph = layerGraph();
lgraph = addLayers(lgraph, rgbBranch);
lgraph = addLayers(lgraph, depthBranch);
lgraph = addLayers(lgraph, mergedLayer);
lgraph = addLayers(lgraph, finalLayers);

% Connect the layers
lgraph = connectLayers(lgraph, 'maxpool2_rgb', 'merge/in1');
lgraph = connectLayers(lgraph, 'maxpool2_depth', 'merge/in2');
lgraph = connectLayers(lgraph, 'merge', 'fc_final');

% Set training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsVali, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
disp("ready to train!!!!");
net = trainNetwork(customDS, lgraph, options);
%% Test Network
[pred, scores] = classify(net, imdsTest);

%% Results
totalTest = testImg*totalPills;
acc = nnz(imdsTest.Labels == pred) / totalTest;
fprintf('Accuracy is %2.2f%%\n', acc*100);

CM = figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    set(gca,'position',[0 0 1 1],'units','normalized');
    cm = confusionchart(imdsTest.Labels, pred); cm.NormalizedValues;
    name = sprintf('CM.png');
    saveas(CM, name);

%% function for reading files
% Function for reading RGB images
function data = readRGBImage(filename)
    [~, label, ~] = fileparts(fileparts(filename));
    %data = {imread(filename), label};
    data = imread(filename);
end

% Function for reading depth images
function data = readDepthImage(filename)
    [~, label, ~] = fileparts(fileparts(filename));
    %data = {imread(filename), label};
    data = imread(filename);
end


