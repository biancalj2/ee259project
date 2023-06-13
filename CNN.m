%% EE 259 Pill Project CNN

clearvars; close all; clc

%% Folder Information
% ===== WHAT YOU CHANGE ===== %
% experimentFolder = sprintf('EE259_Pill_Project'); % folder for experiment (organizes everything in matlab)
dataFolder = sprintf('pills_improved'); % this is the big folder all the data goes in
cnnFolder1 = sprintf('CNN_rgb');
imNum = 1000; % number of images per pill
totalPills = 15; % total number of pills being classified
trainImg = 800; % number of images per pill used for training
testImg = 200; % number of images used per pill for testing
res = 224; % size of images
% =========================== %

% access images
% cd(experimentFolder);
cd(dataFolder);

%% CNN Set Up
imds1 = imageDatastore(cnnFolder1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imds2 = imageDatastore(cnnFolder2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% % labelDsTrain = arrayDatastore(imds1.Labels);
% imdsComplete = combine(imds1, imds2, labelDsTrain);

p = trainImg / imNum;  % percentage of images used for training
[imdsTrain, imdsTest] = splitEachLabel(imds1, p, 'randomized');

[imdsTrain, imdsVali] = splitEachLabel(imdsTrain, (9/10), 'randomized');
[imdsTest, imdsDiscard] = splitEachLabel(imdsTest, testImg, 'randomized');

audsTrain = augmentedImageDatastore([res res 3], imdsTrain);
audsVali  = augmentedImageDatastore([res res 3], imdsVali);
audsTest  = augmentedImageDatastore([res res 3], imdsTest);

% Preview augmented data
minibatch = preview(audsTrain);
sample = figure; imshow(imtile(minibatch.input));
    name = sprintf('sampleData.png');
    saveas(sample, name);

% minibatch_vali = preview(audsVali);
% figure; imshow(imtile(minibatch_vali.input));

% minibatch_test = preview(audsTest);
% figure; imshow(imtile(minibatch_test.input));
% pause();

%% CNN Architecture
fprintf('Creating Network\n');
        
% Prelimary CNN
layers = [
    imageInputLayer([res res 3]); % input layer - images are 224x224
    convolution2dLayer([3 3], 8, 'Stride', 1, 'Padding', 'same');

    batchNormalizationLayer();
    reluLayer();
    maxPooling2dLayer([2 2], 'Stride', 2);
    convolution2dLayer([3 3], 16, 'Stride', 1, 'Padding', 'same', 'NumChannels', 8);

    batchNormalizationLayer();
    reluLayer();
    maxPooling2dLayer([2 2], 'Stride', 2);
    convolution2dLayer([3 3], 32, 'Stride', 1, 'Padding', 'same', 'NumChannels', 16);

    batchNormalizationLayer();
    reluLayer();
    fullyConnectedLayer(totalPills);
    softmaxLayer();
    classificationLayer()
    ];

options = trainingOptions('sgdm', ... 
    'Plots', 'training-progress', ...
    'Momentum', 0.5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', audsVali, ...
    'ValidationPatience', 10, ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 0.01, ...
    'ExecutionEnvironment', 'cpu');

%% Train Network
pillNet = trainNetwork(audsTrain, layers, options); % LG beams

%% Test Network
[pred, scores] = classify(pillNet, audsTest);

%% Results
totalTest = testImg*totalPills;
acc = nnz(imdsTest.Labels == pred) / totalTest;
fprintf('Accuracy is %2.2f%%\n', acc*100);

CM = figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    set(gca,'position',[0 0 1 1],'units','normalized');
    cm = confusionchart(imdsTest.Labels, pred); cm.NormalizedValues;
%     title('Confusion Matrix'); 
    name = sprintf('CM.png');
    saveas(CM, name);


