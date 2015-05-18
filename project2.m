cd 'C:\Users\Acer\Documents\OSU\ST 599_S2015\Project2\ST599pro2'
Train = unzip('train.zip','C:\Users\Acer\Documents\OSU\ST599_S2015\Project2');
fileList = getAllFiles('C:\Users\Acer\Documents\OSU\ST 599_S2015\Project2\Train');
Test = unzip('test.zip','C:\Users\Acer\Documents\OSU\ST599_S2015\Project2');

%gabor filter
%u =5;v =8; m=39; n=39;
%gaborArray=gaborFilterBank(u,v,m,n);
%img = fileList{1}; d1=39 ; d2=39 ;
%featureVector = gaborFeatures(img,gaborArray,d1,d2);

I = imread(Train{57});imshow(I)
points = detectSURFFeatures(I);
[features, valid_points] = extractFeatures(I, points);

figure; imshow(I); hold on;
    plot(valid_points.selectStrongest(10),'showOrientation',true);
    
 [featuress3,validPoints] = extractFeatures(I30336,points) ;
[features,validPoints] = extractFeatures(I,points,Name,Value);


%for loop train
mydata =[];
m = length(Train);
for i = 1:m
    I = imread(Train{i});
    points = detectSURFFeatures(I);
    [features, valid_points] = extractFeatures(I, points);
    mydata=[mydata;mean(features,1)];
end;
%F = mean(features2,1);
%save('mydata');
load('mydata');
x = mydata(~isnan(mydata(:, 1)), :);
save('x');

Test = unzip('test.zip','C:\Users\Acer\Documents\OSU\ST599_S2015\Project2');
%for loop test
mydatatest =[];
l = length(Test);
for i = 1:l
    I = imread(Test{i});
    points = detectSURFFeatures(I);
    [features, valid_points] = extractFeatures(I, points);
    mydatatest=[mydatatest;mean(features,1)];
end;
%F = mean(features2,1);
save('mydatatest');
load('mydatatest');
xt = mydatatest(~isnan(mydatatest(:, 1)), :);
save('xt');



list = dir('train');
cd train;
cd ..
cd train
c1=[891, 15,73,51,18,698,534,244,395,172,817,1936,696,79,683,175,98,180,65,288,108,51,89,32,901,1091,26];
c2=[203,115,44,55,40,57,365,396,916,521,502,38,94,82,90,387,538,98,29,16,138,40,513,12,33,87,116,66,18];
c3=[ 12,129,77,37,231,11,21,25,338,134,14,192,414,276,152,78,705,125,45,58,16,63,16,26,143,110,374,627,1074];
c4=[115,110,15,67,289,160,54,51,155,176, 214,137,485,181,59,249,31,32,130,23,26,40,124,710,56,1981,680,31,441];
c5=[419,354,238,75,319,177,394];
c = [c1 c2 c3 c4 c5];
Y = [];
for i = 1:121
    Y = [Y;i+zeros(c(i),1)];
end;
Y;
u = 30336;
%nan=4251
n = 26085;
ind = randperm(u);
y = Y(ind(1:n));
load('mydata');

analyisData = [x,y];

save('AnalysisData');
x=load('AnalysisData');
load('ex4weights.mat');
%%%%%%%%%%%Add feature
%X = [x x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8 x.^9 x.^10];
load('x'); load('y');
X = x;
m = size(X,1);
%X = [ones(m,1), X];
%%%%%%%
clear ; close all; clc
num_labels = 121;

%%%%OneVsAll
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.8;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  After ...
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%
load('xt');
xt1 = xt(1:100,:);
pred = predictOneVsAll(all_theta, xt1)




  
