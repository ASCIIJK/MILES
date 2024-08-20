clear,clc
train_label_path = "";
test_label_path = "";
save_root = '';
input_root = '';
train_label = load(train_label_path);
test_label = load(test_label_path);
class_name = train_label.class_names;
labels = train_label.annotations;

for i = 1:length(labels)
    current_label = labels(i);
    class_name_path = class_name{current_label.class};
    test = current_label.test;
    if test == 0
        save_folder_path = [save_root, '\train\', class_name_path];
    else
        save_folder_path = [save_root, '\test\', class_name_path];
    end
    if ~exist(save_folder_path, 'dir')
        mkdir(save_folder_path);
    end
    input_path = [input_root, current_label.relative_im_path];
    img = imread(input_path);
    save_path = [save_folder_path, '\' , current_label.relative_im_path(9:end)];
    imwrite(img, save_path);
end