%% EE 259 Pill Project CNN

clearvars; close all; clc

%% Folder Information
% ===== WHAT YOU CHANGE ===== %
% experimentFolder = sprintf('EE259_Pill_Project'); % folder for experiment (organizes everything in matlab)
dataFolder = sprintf('pills_new'); % this is the big folder all the data goes in
cnnFolder = sprintf('CNN_depth'); % or CNN_rgb
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
imds = imageDatastore(cnnFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

p = trainImg / imNum;  % percentage of images used for training
[imdsTrain, imdsTest] = splitEachLabel(imds, p, 'randomized');

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

% UNET
    input = imageInputLayer([res res 1]); % input layer - images are 224x224
    conv1 = convolution2dLayer([3 3], 1, 'Stride', 1, 'Padding', 'same');
    relu1 = reluLayer();
    conv1 = convolution2dLayer([3 3], 1, 'Stride', 1, 'Padding', 'same');
    relu1 = reluLayer('Name', 'relu1');
    maxPool1 = maxPooling2dLayer([2 2], 'Stride', 2);
    pool1 = dropoutLayer(0.25);
    
    conv2 = convolution2dLayer([3 3], 2, 'Stride', 1, 'Padding', 'same');
    relu2 = reluLayer();
    conv2 = convolution2dLayer([3 3], 2, 'Stride', 1, 'Padding', 'same');
    relu2 = reluLayer('Name', 'relu2');
    maxPool2 = maxPooling2dLayer([2 2], 'Stride', 2);
    pool2 = dropoutLayer(0.50);
    
    conv3 = convolution2dLayer([3 3], 4, 'Stride', 1, 'Padding', 'same');
    relu3 = reluLayer();
    conv3 = convolution2dLayer([3 3], 4, 'Stride', 1, 'Padding', 'same');
    relu3 = reluLayer('Name', 'relu3');
    maxPool3 = maxPooling2dLayer([2 2], 'Stride', 2);
    pool3 = dropoutLayer(0.50);
    
    conv4 = convolution2dLayer([3 3], 8, 'Stride', 1, 'Padding', 'same');
    relu4 = reluLayer();
    conv4 = convolution2dLayer([3 3], 8, 'Stride', 1, 'Padding', 'same');
    relu4 = reluLayer('Name', 'relu4');
    maxPool4 = maxPooling2dLayer([2 2], 'Stride', 2);
    pool4 = dropoutLayer(0.50);
    
    convm = convolution2dLayer([3 3], 16, 'Stride', 1, 'Padding', 'same');
    relum = reluLayer();
    convm = convolution2dLayer([3 3], 16, 'Stride', 1, 'Padding', 'same');
    relum = reluLayer('Name', 'relum');
    
    deconv4 = transposedConv2dLayer([3 3], 8, 'Stride', 2, 'Cropping', 'same');
    derelu4 = reluLayer('Name', 'derelu4');
%     concat4 = concatenationLayer(4, 2, 'InputNames', 'derelu4', 'relu4');
    pool5 = dropoutLayer(0.50);
    
    conv6 = convolution2dLayer([3 3], 8, 'Stride', 1, 'Padding', 'same');
    relu6 = reluLayer();
    conv6 = convolution2dLayer([3 3], 8, 'Stride', 1, 'Padding', 'same');
    relu6 = reluLayer();
    
    deconv3 = transposedConv2dLayer([3 3], 4, 'Stride', 2, 'Cropping', 'same');
    derelu3 = reluLayer('Name', 'derelu3');
%     concat3 = concatenationLayer(4, 2, 'derelu3', 'relu3');
    pool6 = dropoutLayer(0.50);
    
    conv7 = convolution2dLayer([3 3], 4, 'Stride', 1, 'Padding', 'same');
    relu7 = reluLayer();
    conv7 = convolution2dLayer([3 3], 4, 'Stride', 1, 'Padding', 'same');
    relu7 = reluLayer();
    
    deconv2 = transposedConv2dLayer([3 3], 2, 'Stride', 2, 'Cropping', 'same');
    derelu2 = reluLayer('Name', 'derelu2');
%     concat2 = concatenationLayer(4, 2, 'derelu2', 'relu2');
    pool7 = dropoutLayer(0.50);
    
    conv8 = convolution2dLayer([3 3], 2, 'Stride', 1, 'Padding', 'same');
    relu8 = reluLayer();
    conv8 = convolution2dLayer([3 3], 2, 'Stride', 1, 'Padding', 'same');
    relu8 = reluLayer();
    
    deconv1 = transposedConv2dLayer([3 3], 1, 'Stride', 2, 'Cropping', 'same');
    derelu1 = reluLayer('Name', 'derelu1');
%     concat1 = concatenationLayer(4, 2, 'derelu1', 'relu1');
    pool8 = dropoutLayer(0.50);
    
    conv9 = convolution2dLayer([3 3], 1, 'Stride', 1, 'Padding', 'same');
    relu9 = reluLayer();
    conv9 = convolution2dLayer([3 3], 1, 'Stride', 1, 'Padding', 'same');
    relu9 = reluLayer();
    
    fullyConnect = fullyConnectedLayer(totalPills);
    softmax = softmaxLayer();
    classify = classificationLayer();

UNet = [input; 
            conv1; relu1; conv1; relu1; maxPool1; pool1;
            conv2; relu2; conv2; relu2; maxPool2; pool2;
            conv3; relu3; conv3; relu3; maxPool3; pool3;
            conv4; relu4; conv4; relu4; maxPool4; pool4;
            convm; relum; convm; relum;
            deconv4; derelu4; %concat4; 
            pool5;
            conv6; relu6; conv6; relu6;
            deconv3; derelu3; %concat3; 
            pool6;
            conv7; relu7; conv7; relu7;
            deconv2; derelu2; %concat2; 
            pool7;
            conv8; relu8; conv8; relu8;
            deconv1; derelu1; %concat1; 
            pool8;
            conv9; relu9; conv9; relu9;
            fullyConnect; softmax; classify];

options = trainingOptions('sgdm', ... 
    'Plots', 'training-progress', ...
    'Momentum', 0.5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', audsVali, ...
    'ValidationPatience', 20, ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.01, ...
    'ExecutionEnvironment', 'cpu');

%% Train Network
pillNet = trainNetwork(audsTrain, UNet, options); % LG beams

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














