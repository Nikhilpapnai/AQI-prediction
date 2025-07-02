% Load dataset
data = readtable(State_weather_aqi_data2.csv');

% Extract inputs and output
X = table2array(data(:, 1:12));
Y = table2array(data(:, 13));

% One-hot encode state, city, station (cols 1, 2, 3)
state = dummyvar(categorical(X(:,1)));
city = dummyvar(categorical(X(:,2)));
station = dummyvar(categorical(X(:,3)));
numeric_features = X(:, 6:12);

features = [state city station numeric_features];

% Split data
cv = cvpartition(size(features,1),'HoldOut',0.25);
idx = cv.test;
XTrain = features(~idx,:);
YTrain = Y(~idx);
XTest = features(idx,:);
YTest = Y(idx);

%--- Linear Regression ---
mdl_lr = fitlm(XTrain, YTrain);
Ypred_lr = predict(mdl_lr, XTest);

%--- Polynomial Regression (degree 2) ---
XTrain_poly = [XTrain XTrain.^2];
XTest_poly = [XTest XTest.^2];
mdl_pr = fitlm(XTrain_poly, YTrain);
Ypred_pr = predict(mdl_pr, XTest_poly);

%--- Decision Tree Regression ---
mdl_dt = fitrtree(XTrain, YTrain);
Ypred_dt = predict(mdl_dt, XTest);

%--- Random Forest Regression ---
mdl_rf = TreeBagger(500, XTrain, YTrain, 'Method', 'regression');
Ypred_rf = predict(mdl_rf, XTest);

%--- Support Vector Regression (scaled) ---
XTrain_s = zscore(XTrain);
XTest_s = (XTest - mean(XTrain)) ./ std(XTrain);
mdl_svr = fitrsvm(XTrain_s, YTrain, 'KernelFunction', 'rbf');
Ypred_svr = predict(mdl_svr, XTest_s);

%--- Evaluation Functions ---
rmse = @(a,b) sqrt(mean((a - b).^2));
mae = @(a,b) mean(abs(a - b));
r2 = @(a,b) 1 - sum((a - b).^2) / sum((a - mean(a)).^2);
rmsle = @(a,b) sqrt(mean((log(b+1) - log(a+1)).^2));

% --- Evaluation on test data ---
fprintf('\nModel\t\tR^2\t\tRMSE\t\tMAE\t\tRMSLE\n');
fprintf('Linear\t\t%.4f\t%.4f\t%.4f\t%.4f\n', r2(YTest,Ypred_lr), rmse(YTest,Ypred_lr), mae(YTest,Ypred_lr), rmsle(YTest,Ypred_lr));
fprintf('PolyReg\t\t%.4f\t%.4f\t%.4f\t%.4f\n', r2(YTest,Ypred_pr), rmse(YTest,Ypred_pr), mae(YTest,Ypred_pr), rmsle(YTest,Ypred_pr));
fprintf('DecisionTree\t%.4f\t%.4f\t%.4f\t%.4f\n', r2(YTest,Ypred_dt), rmse(YTest,Ypred_dt), mae(YTest,Ypred_dt), rmsle(YTest,Ypred_dt));
fprintf('RandomForest\t%.4f\t%.4f\t%.4f\t%.4f\n', r2(YTest,Ypred_rf), rmse(YTest,Ypred_rf), mae(YTest,Ypred_rf), rmsle(YTest,Ypred_rf));
fprintf('SVR\t\t%.4f\t%.4f\t%.4f\t%.4f\n', r2(YTest,Ypred_svr), rmse(YTest,Ypred_svr), mae(YTest,Ypred_svr), rmsle(YTest,Ypred_svr));
