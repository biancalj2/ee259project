%% EE 259 Image Processor

clearvars; close all; clc

%% Parameters
% ===== WHAT YOU CHANGE ===== %
% 
experimentFolder = sprintf('EE_259_CNN'); % folder for experiment (organizes everything in matlab)
dataFolder = sprintf('pills_newRGB_1000_raw'); % this is the big folder all the data goes in
CNNFolder = sprintf('CNN_rgb');   
pillTotal = 15;
numImg = 1000;
cnn_img_size = 224; % pixels - size of square image for CNN
viewNum = 0; % lets you visualize on the crop beam # entered
LOW_IN = 0.10; HIGH_IN = 0.90; LOW_OUT = 0; HIGH_OUT = 1; % parameters to control imadjust
% =========================== %

%%
% cd(experimentFolder);
cd(dataFolder);

% creates folder for CNN data
mkdir(CNNFolder);
cd(CNNFolder);

for i = 1:pillTotal
    newFolder = sprintf('Pill_%02d', i);
    mkdir(newFolder);
end

cd ..

folder = 1;
idxImg = 1;

dataFileName = sprintf('pill_%d/rgb_%d.jpg', folder, idxImg);

px_val = imread(dataFileName); % i needs to be an int here
figure; imshow(px_val); % title('Unprocessed')

% Adjust contrast - only for depth images
% px_val = imadjust(px_val, [LOW_IN; HIGH_IN], [LOW_OUT; HIGH_OUT]); % adjust contrast between beam and background

% Resize
pillImg = imresize(px_val, [cnn_img_size cnn_img_size], 'Colormap', 'original'); % resizes image to input for CNN

figure; imshow(pillImg); title('Processed') % show processed image

pause();

for i = 1:pillTotal
    folderName = sprintf('/Pill_%02d', i); % folder to put images in
    folderName = strcat(CNNFolder, folderName);
       
    for j = 1:numImg
        dataFileName = sprintf('pill_%d/rgb_%d.jpg', i, j);
        px_val = imread(dataFileName); % i needs to be an int here
%         px_val = imadjust(px_val, [LOW_IN; HIGH_IN], [LOW_OUT; HIGH_OUT]); % adjust contrast between beam and background
        pillImg = imresize(px_val, [cnn_img_size cnn_img_size], 'Colormap', 'original'); % resizes image to input for CNN
        
        imgFileName = sprintf('rgb_%02d.png', j);
        fullFileName = fullfile(folderName, imgFileName);
        imwrite(pillImg, fullFileName);
        fprintf('%s_img%d\n', folderName, j)
    end
end

cd ../..



