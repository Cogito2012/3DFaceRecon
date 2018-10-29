function [ all_labels ] = parse_vggface_labels( labelfile )

all_labels = [];
fid = fopen(labelfile, 'r');
ind = 0;
while ~feof(fid)
    ind = ind + 1;
    line_data = regexp(fgetl(fid), '\s+', 'split');
    all_labels.imgIDs{ind} = line_data{1};
    all_labels.imgURLs{ind} = line_data{2};
    all_labels.rectbox{ind} = [str2num(line_data{3}), str2num(line_data{4}), str2num(line_data{5}), str2num(line_data{6})];  % [left top right bottom]
    all_labels.poseIDs(ind) = str2num(line_data{7});
    all_labels.dpmScores(ind) = str2num(line_data{8});
    all_labels.curation(ind) = str2num(line_data{9});
end
fclose(fid);

end

