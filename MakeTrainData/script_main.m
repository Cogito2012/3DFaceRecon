clc
clear
close all
addpath(genpath('./ZBuffer'))
%%
dataset_path = '../../data/VGGFace/vgg_face_dataset';
images_path = fullfile(dataset_path, 'images');
srcfiles_path = fullfile(dataset_path, 'files');
% prepare output paths
output_path = '../../data/VGGFace/train';
faces_path = fullfile(dataset_path, 'face_images');
if ~exist(faces_path, 'dir')
    mkdir(faces_path);
end
pnccs_path = fullfile(output_path, 'pnccs');
if ~exist(pnccs_path, 'dir')
    mkdir(pnccs_path);
end
maskims_path = fullfile(output_path, 'mask_images');
if ~exist(maskims_path, 'dir')
    mkdir(maskims_path);
end
labels_path = fullfile(output_path, 'labels');
if ~exist(labels_path, 'dir')
    mkdir(labels_path);
end
list_file_id = fopen(fullfile(output_path, 'filelist.txt'), 'w');
im_size = 200;
beta = 0.7;
%% load morphable models
vertex_data = load('../3dmm/vertex_code.mat');
shape_model = load('../3dmm/Model_Shape.mat');
expression_model = load('../3dmm/Model_Expression.mat');
vertex_code = vertex_data.vertex_code;
mu_shape = shape_model.mu_shape;
w = shape_model.w;
tri = shape_model.tri;
% sigma = shape_model.sigma;
mu_exp = expression_model.mu_exp;
w_exp = expression_model.w_exp;
% sigma_exp = expression_model.sigma_exp;
num_pose_param = 7;
num_shape_param = size(w, 2);
num_exp_param = size(w_exp, 2);
mu = mu_shape + mu_exp;

%% open the parallel environment
tic
pool = start_matlabpool(4);
toc
%%
subjects = dir(images_path);
subjects = subjects(3:end);
num_subjects = length(subjects);
for i=1:num_subjects
    subj_name = subjects(i).name;
%     if ~strcmp(subj_name, 'Abel_Ferrara')
%         continue;
%     end
    labelfile = fullfile(srcfiles_path, [subj_name, '.txt']);
    try 
        all_labels = parse_vggface_labels(labelfile);
    catch
        continue;
    end
    % prepare subjects' folder
    faces_subj_path = fullfile(faces_path, subj_name);
    if ~exist(faces_subj_path, 'dir')
        mkdir(faces_subj_path);
    end
    pnccs_subj_path = fullfile(pnccs_path, subj_name);
    if ~exist(pnccs_subj_path, 'dir')
        mkdir(pnccs_subj_path);
    end
    maskims_subj_path = fullfile(maskims_path, subj_name);
    if ~exist(maskims_subj_path, 'dir')
        mkdir(maskims_subj_path);
    end
    labels_subj_path = fullfile(labels_path, subj_name);
    if ~exist(labels_subj_path, 'dir')
        mkdir(labels_subj_path);
    end
    % processing images for each subject
    all_images = dir(fullfile(images_path, subj_name, '*.jpg'));
    fprintf('Processing images of subject:%s (Total: %d)...', subj_name, length(all_images));
    tic
    parfor j=1:length(all_images)
        imgfile = fullfile(images_path, subj_name, all_images(j).name);
        imgbytes = all_images(j).bytes;
        [im, msg] = read_image_file(imgfile, imgbytes);
        if ~msg, continue; end
        [height, width, nchannels] = size(im);
        % find the labels for current image
        imgID = all_images(j).name(1:end-4);
%         disp([num2str(j), ': ', imgID])
%         if ~strcmp(imgID, '00000873')
%             continue;
%         end
        idx = find(strcmp(all_labels.imgIDs, imgID));
        rectbox = all_labels.rectbox{idx};
        rectbox = check_box(rectbox, height, width);
        im_face = im(rectbox(2):rectbox(4), rectbox(1):rectbox(3), :);
        im_face = imresize(im_face, [im_size, im_size]);
        % 
        [ pose_param, shape_param, exp_param ] = get_random_params( im_size, num_shape_param, num_exp_param, beta);
        [pncc, masked_img] = Project2D(mu, w, w_exp, tri, vertex_code, im_face, pose_param, shape_param, exp_param);
        % save results
        faceim_file = fullfile(faces_subj_path, [imgID,'.jpg']);
        imwrite(im_face, faceim_file);
        pncc_file = fullfile(pnccs_subj_path, [imgID,'.jpg']);
        imwrite(pncc, pncc_file);
        maskim_file = fullfile(maskims_subj_path, [imgID,'.jpg']);
        imwrite(masked_img, maskim_file);
        labelfile = fullfile(labels_subj_path, [imgID,'.txt']);
        f = fopen(labelfile, 'wt');
        for k=1:num_pose_param
            fprintf(f, '%.6f\n', pose_param(k));
        end
        for k=1:num_shape_param
            fprintf(f, '%.6f\n', shape_param(k));
        end
        for k=1:num_exp_param
            fprintf(f, '%.6f\n', exp_param(k));
        end
        fclose(f);
%         fprintf(list_file_id, '%s/%s\n', subj_name, imgID);
    end
    t = toc;
    fprintf('Time: %d seconds.\n', t);
end
close_matlabpool;

%% We seperately write the list file here is due to the parallel processing above.
valid_data = dir(labels_path);
valid_data = valid_data(3:end);
for i=1:length(valid_data)
    subj_name = valid_data(i).name;
    all_labels = dir(fullfile(labels_path, subj_name, '*.txt'));
    for j=1:length(all_labels)
        imgID = all_labels(j).name;
        fprintf(list_file_id, '%s/%s\n', subj_name, imgID);
    end
end
fclose(list_file_id);



