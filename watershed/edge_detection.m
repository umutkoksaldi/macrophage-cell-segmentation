I = imread('first_pass_back_ext_res/jw-24h 1_c1/jw-24h 1_c1-14.png');

[bw, threshold] = edge(I, 'Canny', 0.2);

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);

figure, imshow(gradmag, []);
% bw = bwareaopen(bw, 50);
figure, imshow(imoverlay(I, bw));
