% Rewrite session file and create ground truth data source
%
% Copyright 2018 The MathWorks, Inc.

cd images;

load('imageLabelingSession.mat')
imageLabelingSession.FileName = fullfile(pwd, 'imageLabelingSession.mat');
sz = size(imageLabelingSession.ImageFilenames);
for i = 1:sz(1)
    fname = imageLabelingSession.ImageFilenames{i};
    [pathstr,name,ext] = fileparts(fname);
    fname = fullfile(pwd, [name ext]);
    imageLabelingSession.ImageFilenames{i} = fname;
end

save('imageLabelingSession.mat', 'imageLabelingSession')

folderInfo = dir('train/ground_truth/labeled');
folderInfo = folderInfo(~ismember({folderInfo.name}, {'.', '..'}));
sz = size({folderInfo.name});
idx = 1;
for i = 1:sz(2)
    [pathstr,name,ext] = fileparts(fullfile(folderInfo(i).folder, folderInfo(i).name));
    if strcmp(ext, '.png')
        labellist{idx, 1} = fullfile(pathstr, [name ext]);
        idx = idx + 1;
    end
end


folderInfo = dir('train/*.png');
folderInfo = folderInfo(~ismember({folderInfo.name}, {'.', '..'}));
sz = size({folderInfo.name});
idx = 1;
for i = 1:sz(2)
    [pathstr,name,ext] = fileparts(fullfile(folderInfo(i).folder, folderInfo(i).name));
    if strcmp(ext, '.png')
        imagelist{idx, 1} = [name ext];
        idx = idx + 1;
    end
end

dataSource = groundTruthDataSource(imagelist);

names = {'Background';'Cells'};
types = [labelType('PixelLabel');labelType('PixelLabel')];
pixelLabelID = {1;2};
labelDefs = table(names,types, pixelLabelID, ...
                      'VariableNames',{'Name','Type','PixelLabelID'})
labelData = table(labellist,'VariableNames',{'PixelLabelData'})
gTruth = groundTruth(dataSource,labelDefs,labelData)

cd ../


