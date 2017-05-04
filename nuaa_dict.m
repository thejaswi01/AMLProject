load('deep_dict_learn_nuaa.mat');
load('nuaa_data.mat');

train_images = Wtr;
train_labels = labels';

test_images = Wte;
test_labels = tlabels';

%train_images = train_images';
%test_images = test_images';
% display_network(train_images(:,1:100)); % Show the first 100 images
% disp(train_labels(1:10));


% KNN
tic;
knn_mdl = fitcknn(train_images, train_labels, 'NumNeighbors', 5, 'Standardize', 1, 'Distance', 'euclidean', 'NSMethod', 'kdtree');
end_time = toc;

predict_labels = predict(knn_mdl, test_images);

knn_acc = sum(predict_labels == test_labels) / size(test_labels, 1) 
fprintf('\n K-NN Accuracy: %f ', knn_acc * 100);
fprintf('\n K-NN Error: %f ', (1 - knn_acc) * 100);
fprintf('\n K-NN Training time: %f ', end_time);


% SVM
tic;
svm_mdl = fitcsvm(train_images, train_labels);
end_time = toc;
%save('./svm_mdl', svm_mdl);
predict_labels = predict(svm_mdl, test_images);

svm_acc = sum(predict_labels == test_labels) / size(test_labels, 1) 
fprintf('\n SVM Accuracy: %f ', svm_acc * 100);
fprintf('\n SVM Error: %f ', (1 - svm_acc) * 100);
fprintf('\n SVM Training time: %f ', end_time);
