clc, clear, close all
trainData = importdata('Train.csv');
trainData = mixData(trainData);
inputsTrain = trainData(:,9:end);
targetsTrain = trainData(:,1:8);


testData =  importdata('Test.csv');

inputsTest = testData(:,9:end);
targetsTest = testData(:,1:8);



% patternnet set up
hiddenLayerSize = [10 10 10];
net = patternnet(hiddenLayerSize);
net.trainParam.epochs = 100000;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% train
[net,tr] = train(net,inputsTrain',targetsTrain');

outputs = net(inputsTest')';
outputs = round(outputs);
accuracy = 1- errRate(targetsTest, outputs);

est_y = getIndex(outputs);
des_y = getIndex(targetsTest);

figure
hold on
plot(est_y,'ob')
plot(des_y, '--r')
legend('predicted', 'acutal')



