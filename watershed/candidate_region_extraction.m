% read necessary files
filename = 'jw-24h 1_c1';
I = imread(sprintf('images/%s.png', filename));
corr = imread('images/other/jw-24h 1_c5.png');
original = I;
gt = imread(sprintf('images/ground_truth/%s gt.png', filename));

% convert yellow to grayscale and histogram adaptation
I = I(:,:,2);
I_eq = adapthisteq(I);

% convert yellow to binary image
bw = imbinarize(I_eq, adaptthresh(I_eq));
bw = imfill(bw, 'holes');
bw = bwareaopen(bw, 200);

% convert blue cells to binary image
bw_corr = imbinarize(adapthisteq(rgb2gray(corr)), adaptthresh(adapthisteq(rgb2gray(corr))));
bw_corr = bwareaopen(bw_corr, 500);
bw_corr = imclearborder(bw_corr);
bw_corr = imclose(bw_corr, ones(6, 6));

figure
imshow(bw_corr);

figure
imshow(imoverlay(original,bw)), title('black and white');

% complement the image and eliminate the completely white background parts
I_eq = imcomplement(I_eq);
I_eq = mask(I_eq, bw);
I_eq = adapthisteq(I_eq);

figure
imshow(I_eq);

I_eq_c = imcomplement(I_eq);
I_mod = imimposemin(I_eq_c, bw_corr);
        
L = watershed(I_mod);
labeled_image = label2rgb(L);
labeled_image = rgb2gray(labeled_image);
labeled_image(labeled_image < 255) = 0;
bw_labeled = logical(labeled_image);

figure, imshow(labeled_image), title('Labeled image');

figure, imshow(imoverlay(gt, bw_labeled));
figure, imshow(imoverlay(I, bw_labeled));
% extract candidate regions
for i=(min(min(L))+1):max(max(L))
    ext_mask = L;
    ext_mask(ext_mask ~= i) = 0;
    ext_mask(ext_mask == i) = 1;
    ext_mask = logical(ext_mask);
    
    img_to_save = mask(I, ext_mask);
    imwrite(img_to_save, sprintf('first_pass_back_ext_res/%s/%s-%d.png', filename, filename, i));
end



function I_m = mask(I, mask_array)
    I_m = I.*uint8(mask_array);
end

function [BW,maskedImage] = activeContour(X)
    
    % Threshold image - adaptive threshold
    BW = imbinarize(X, 'adaptive', 'Sensitivity', 0.880000, 'ForegroundPolarity', 'bright');
    
    % Active contour
    iterations = 100;
    BW = activecontour(X, BW, iterations, 'edge');
    
    % Create masked image.
    maskedImage = X;
    maskedImage(~BW) = 0;
end

