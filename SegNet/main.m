
clear all, close all, clc;

vgg16();


% load images

imgDir = fullfile(pwd,'images/train');
imds = imageDatastore(imgDir);

I = readimage(imds, 1);
imshow(I)

% Label Ground Truth in a Collection of Images
reconstLabelSession


imageLabeler;

% Load Pixel-Labeled Images

classes = [
    "Background"
    "Cells"
    ];

pixelLabelID = cell(2,1);
pixelLabelID{1,1} = 1;
pixelLabelID{2,1} = 2;


labelDir = fullfile(imgDir,'ground_truth/labeled');
pxds = pixelLabelDatastore(labelDir,classes,pixelLabelID);


C = readimage(pxds, 1);

cmap = bloodSmearColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);

figure
imshow(B)
pixelLabelColorbar(cmap,classes);

% Analyze Dataset Statistics

tbl = countEachLabel(pxds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')


% Resize Images

imageFolder = fullfile(imgDir,'imagesReszed',filesep);
imds = resizeBloodSmearImages(imds,imageFolder);

labelFolder = fullfile(labelDir,'labelsResized',filesep);
pxds = resizeBloodSmearPixelLabels(pxds,labelFolder);

% Prepare Training and Test Sets

[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionBloodSmearData(imds,pxds);

numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

% Create the Network

imageSize = [300 300 3];
numClasses = numel(classes);
lgraph = segnetLayers(imageSize,numClasses,'vgg16');

% Balance Classes Using Class Weighting

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq


pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

% SegNet ipixel classification layer

lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');

figure, plot(lgraph)

% Select Training Options
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-2, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 30, ...  
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency', 1000);

% Data Augmentation
augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation', [-10 10], 'RandRotation', [-180 180]);

%% Start Training
datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
   'DataAugmentation',augmenter);


[net, info] = trainNetwork(datasource,lgraph,options);

% Test Network on One Image

idx = 1;
I = readimage(imdsTest,idx);
C = semanticseg(I, net);

B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure, imshowpair(I, B, 'montage');
pixelLabelColorbar(cmap, classes);

expectedResult = readimage(pxdsTest,idx);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

iou = jaccard(C, expectedResult);
table(classes,iou)



pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);


metrics.DataSetMetrics

metrics.ClassMetrics

%% Supporting Functions

function pixelLabelColorbar(cmap, classNames)
colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end
%% 

function cmap = bloodSmearColorMap()

cmap = [
    128 128 128
    000 000 192
    ];


cmap = cmap ./ 255;
end


function imds = resizeBloodSmearImages(imds, imageFolder)

if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);     
    
    % Resize image.
    I = imresize(I,[300 300]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end
%% 
% 

function pxds = resizeBloodSmearPixelLabels(pxds, labelFolder)
% Resize pixel label data to [300 300].

classes = pxds.ClassNames;
labelIDs = 1:numel(classes);
if ~exist(labelFolder,'dir')
    mkdir(labelFolder)
else
    pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
    return; % Skip if images already resized
end

reset(pxds)
while hasdata(pxds)
    % Read the pixel data.
    [C,info] = read(pxds);
    
    % Convert from categorical to uint8.
    L = uint8(C);
    
    % Resize the data. Use 'nearest' interpolation to
    % preserve label IDs.
    L = imresize(L,[300 300],'nearest');
    
    % Write the data to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(L,[labelFolder filename ext])
end

labelIDs = 1:numel(classes);
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
end
%% 
% 

function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionBloodSmearData(imds,pxds)
% Partition Blood Smear Images by randomly selecting 95% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(25); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 95% of the images for training.
N = round(0.95 * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end