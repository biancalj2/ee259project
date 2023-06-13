%% CNN Tester
% Will Jarrett
% must have a CNN trained and loaded into the workspace (called beamNet)

%% Testing Image Set    
% ===== WHAT YOU CHANGE ===== %
% experimentFolder = sprintf('EE259_Pill_Project'); % folder for experiment (organizes everything in matlab)
dataFolder = sprintf('pills_unimproved'); % this is the big folder all the data goes in
cnnFolder = sprintf('CNN_depth');
totalPills = 15;
testImg = 200; % number of images per pill for testing
imgSize = 224; % size of images
color = 3; % number of color channels in images
% =========================== %

% access images
% cd(experimentFolder);
cd(dataFolder);

%% Image Datastore Set Up
imds_2 = imageDatastore(cnnFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTesting, imdsExtra] = splitEachLabel(imds_2, testImg, 'randomized');
audsTesting  = augmentedImageDatastore([imgSize imgSize color], imdsTesting);

%% Test Network
tic
[pred_test, scores_test] = classify(pillNet, audsTesting);
toc
elapsed_minutes = floor(toc/60);
elapsed_seconds = round(((toc/60) - floor(toc/60))*60, 0);
fprintf('Total time to test %d images on CNN: %d:%d\n', trainImg*totalPills, elapsed_minutes, elapsed_seconds);

%% Results
totalTest_2 = totalPills*testImg;
acc_test = nnz(imdsTesting.Labels == pred_test)/(totalTest_2);
figure; cm_test = confusionchart(imdsTesting.Labels, pred_test);
title('Training Improved Depth, Testing Unimproved Depth')
cm_test.NormalizedValues

name = sprintf('CMs/CM_dep2_dep1.png');
saveas(cm_test, name);

fprintf('Total time to test %d images on CNN: %d:%d\n', testImg*totalPills, elapsed_minutes, elapsed_seconds);
fprintf('Accuracy is %2.2f%%\n', acc_test*100);
    
