%% Shape Detection and Image Processing for Pill Sorting 
clc; close all; clear all;

%% ~~~~~~~~~~~~~~~~~~~~~~~~~ Shape Detection ~~~~~~~~~~~~~~~~~~~~~~~~~
% Tuning Parameters 
low_intensity = 78/256; % low end intensity contrast mapping value
high_intensity = 155/256; % high end intensity contrast mapping value
sensitivity_factor = 0.64; % sensitivity factor for adaptive thresholding

% Testing shape_sort function for all pills 
test = zeros(15, 1001); % shape detection results matrix
for j = 1:15 % pill numbers
    pill_number = int2str(j);
    for i = 0:1000 % image numbers 
        depth_img = imread("pills/pill_" + pill_number + "/depth_" + int2str(i) + ".jpg"); % reads image
        shape = shape_sort(depth_img, low_intensity, high_intensity,...
            sensitivity_factor); % 1 for oval/tube, 2 for circle, 3 for undetermined
        test(j, i+1) = shape;
        
        if shape == 1 % classified as oval/tube
            depth_img = insertMarker(depth_img,[70 320],'x','color','black','size',30); % adding x mark to image
        elseif shape == 2 % classified as circle
            depth_img = insertMarker(depth_img,[70 320],'s','color','black','size',30); % adding square mark to image
        else % uncertain classification 
            depth_img; % image remains unchanged
        end  
        %imwrite(I_depth ,"pills_new_1000/pill_" + pill_number + "/depth_" + int2str(i) + ".jpg"); % saving modified depth images
    end
end

%% ~~~~~~~~~~~~~~~~~~~~~~~~~ RGB Image Processing ~~~~~~~~~~~~~~~~~~~~~~~~~ 
% Tuning Parameters
edge_threshold = 0.1; % amplitude of strong edges to leave intact for contrast
smoothing_contrast = 1; % amount of enhancement or smoothing desired for contrast
smoothing_bilateral = 1000; % amount of smoothing for bilateral filtering
sigma_spatial = 3; % standard deviation of spatial Guassian smoothing kernel
sigma_gaussian = 0.5; % standard deviation of Gaussian low-pass filter
sharpening_effect = 0.7; % strength of sharpening effect

for j = 1:15 % pill numbers
    pill_number = int2str(j);
    for i = 0:1000 % image numbers
        rgb_img = imread("pills/pill_" + pill_number + "/rgb_" + int2str(i) + ".jpg"); % reads image
        rgb_img = imsharpen(imbilatfilt(localcontrast(rgb_img,edge_threshold,smoothing_contrast)...
            ,smoothing_bilateral,sigma_spatial),'Radius',sigma_gaussian,'Amount',sharpening_effect); % image processing
        %imwrite(rgb_img ,"pills_new_1000/pill_" + pill_number + "/rgb_" + int2str(i) + ".jpg"); % saving modified rgb images
    end
end

%% Creating Random Depth and RGB Images 51 through 1000 for 15 Pills
% for j = 1:15
%     pill_number = int2str(j);
%     for i = 51:1000
%         img_number = randi([0 50]); % randomly chooses one of the original 51 images
%         
%         [I_depth, I_rgb] = new_image("pills/pill_" + pill_number + "/depth_" + int2str(img_number) + ".jpg",...
%             "pills/pill_" + pill_number + "/rgb_" + int2str(img_number) + ".jpg");
%         imwrite(I_depth ,"pills/pill_" + pill_number + "/depth_" + int2str(i) + ".jpg"); % saving additional depth images
%         imwrite(I_rgb ,"pills/pill_" + pill_number + "/rgb_" + int2str(i) + ".jpg"); % saving additional rgb images
%     end
% end

%% Functions
function [I_depth, I_rgb] = new_image(depth_img, rgb_img)
% randomizes existing pill image 
    r1 = randi([0 360]); % random rotation variable
    r2 = randi([1 4]); % random reflection variable
    r3 = 1 + 5*rand; % random noise variable 

    I = imread(depth_img); % initial image depth
    I_depth = imrotate(I, r1, "crop"); % rotating
    I_depth = mirror(I_depth, r2); % mirroring/reflecting
    I_depth = imgaussfilt(I_depth, r3); % darkening 
    
    I2 = imread(rgb_img); % initial image rgb
    I_rgb = imrotate(I2, r1, "crop"); % rotating
    I_rgb = mirror(I_rgb, r2); % mirroring/reflecting
    I_rgb = imgaussfilt(I_rgb, r3); % blurring  
end

function I_mirror = mirror(I, random)
% this function randomly flips/mirrors an image     
    if random == 1
         I_mirror = I; % image stays the same
    else if random == 2
        I_mirror = flipdim(I, 2); % horizontal flip
        else if random == 3
                I_mirror = flipdim(I, 1); % vertical flip
            else
                I_mirror = flipdim(flipdim(I, 1), 2); % horizontal and vertical flip
            end
        end
    end
end

function shape = shape_sort(image, low, high, factor)
% this function takes a depth image and outputs a pill shape classification

    % Narrows down pill filtering options based on depth images 
    depth_image_original = image; % initial image
    depth_image = imcrop(depth_image_original,[110 110 420 420]); % cropping image
    
    binary_image = imbinarize(imadjust(depth_image,[low high]),'adaptive','Sensitivity',factor); % binary and contrast to isolate pill
    test_image = imbinarize(imadjust(depth_image,[120/256 200/256]),'adaptive','Sensitivity',0.61); % repeating with test comparison parameters
    
    binary_image_flip = bwareafilt(imcomplement(binary_image),2); % complement of image and filters out two largest objects
    test_flip = bwareafilt(imcomplement(test_image),1); % repeating for test comparison case
    
    figure()
    tiledlayout(2,2)
    
    nexttile
    imshow(depth_image_original)
    title('Original Depth Image')
    
    nexttile
    imshow(binary_image)
    title('Binary Depth Image')
    
    nexttile
    imshow(binary_image_flip)
    title('Complement Isolated Objects')
    
    nexttile
    
    % finding object properties
    s = regionprops(binary_image_flip,{...
        'Centroid',...
        'MajorAxisLength',...
        'MinorAxisLength',...
        'Orientation',...
        'Circularity',...
        'Area'});
    
    % object properties for test comparison
    s_test = regionprops(test_flip,{...
        'Centroid',...
        'MajorAxisLength',...
        'MinorAxisLength',...
        'Orientation',...
        'Circularity',...
        'Area'});
    
    imshow(imcomplement(binary_image_flip),'InitialMagnification','fit')
    
    t = linspace(0,2*pi,50);
    
    hold on
    % plotting best fit ellipse
    for k = 1:length(s)
        a = s(k).MajorAxisLength/2;
        b = s(k).MinorAxisLength/2;
        Xc = s(k).Centroid(1);
        Yc = s(k).Centroid(2);
        phi = deg2rad(-s(k).Orientation);
        x = Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
        y = Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);
        plot(x,y,'r','Linewidth',3)
    end
    
    hold off
    
    title('Best Fit Ellipse')
   
    [argvalue, argmax] = max([s(1).Area, s(2).Area]); % finding index of largest object
    
    if (s(argmax).MajorAxisLength/s(argmax).MinorAxisLength >= 1.35 && ...
            s_test.MajorAxisLength/s_test.MinorAxisLength < 1.35) || ...
            (s(argmax).MajorAxisLength/s(argmax).MinorAxisLength <= 1.25 && ...
            s_test.MajorAxisLength/s_test.MinorAxisLength > 1.25) % accounting for table being too significant and/or close to pill
        disp("Can't be determined")
        shape = 3;
    else  
        if max([s(1).Area, s(2).Area])/min([s(1).Area, s(2).Area]) > 2.5 % insignificant table impact so pill dominates image
            if s(argmax).MajorAxisLength/s(argmax).MinorAxisLength >= 1.35
                disp("Oval or Tube")
                shape = 1;
            elseif s(argmax).MajorAxisLength/s(argmax).MinorAxisLength <= 1.25
                    disp("Circular")
                    shape = 2;
                else
                    disp("Can't be determined")
                    shape = 3;     
            end
        else % significant component of table in image that passes first check
            if s(1).MajorAxisLength/s(1).MinorAxisLength >= 1.35 && ...
                    s(2).MajorAxisLength/s(2).MinorAxisLength >= 1.35 % both objects oval/tube
                disp("Oval or Tube")
                disp(index)
                shape = 1;
            elseif s(1).MajorAxisLength/s(1).MinorAxisLength <= 1.25 && ...
                        s(2).MajorAxisLength/s(2).MinorAxisLength <= 1.25 % both objects circular
                    disp("Circular")
                    disp(index)
                    shape = 2;
                else
                    disp("Can't be determined")
                    disp(index)
                    shape = 3;
            end
        end
    end
end