directory = dir('images/train/ground_truth');

for i=3:22
   filename = directory(i).name;
   I = imread(strcat('images/train/ground_truth/', filename));
   I = rgb2gray(I);
   I(I > 0) = 1;
   bw = logical(I);
   % bw = imdilate(bw, strel('diamond', 5));
   bw = imfill(bw, 'holes');
   I(bw == 1) = 2;
   I(bw == 0) = 1;
   imwrite(I, strcat('images/train/ground_truth/labeled/', strrep(filename, 'gt', 'labeled')));
end

