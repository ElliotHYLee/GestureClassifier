err2 = 100;
err1 = 100;
while(err1 > 2)
    clc, clear, close all

    %% just preparing data set
    trainData = importdata('TrainSet.mat');
    testData = importdata('TestSet.mat');

    [row, col] = size(trainData);
    trainSet = round(trainData(:,1:col-1)/2*1000);
    intputDim = col;
    trainLabel = trainData(:,col);

    testSet = round(testData(:,1:col-1)/2*1000);
    testLabel = testData(:,col);

    targetsTest = zeros(row, 8);
    targetsTrain = zeros(row, 8);
    for i=1:1:row
       temp = trainLabel(i);
       targetsTrain(i,temp) = 1;
       temp = testLabel(i);
       targetsTest(i,temp) = 1;
    end
    inputsTrain = trainSet';
    targetsTrain = targetsTrain';

    %% finaly, i got the data set for train function
    inputsTest = testSet';
    targetsTest = targetsTest';

    % patternnet set up
    hiddenLayerSize = 10;
    net = patternnet([hiddenLayerSize]);
    net.trainParam.epochs = 5000;
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    % train
    [net,tr] = train(net,inputsTrain,targetsTrain);
    
    outputs = net(inputsTest);
    class = (vec2ind(outputs))';
    err1 = errRate(class, testLabel)
    
    
    layer1 = net.IW{1,1};
    layer2 = net.LW{2,1};
    layer1Bias = net.b{1,1};
    layer2Bias = net.b{2,1};

    % combine weight vectors with bias.
    layer11 = layer1';
    layer11 = [layer1Bias'; layer11];
    layer11 = layer11';
    % each row is a perceptron, first column is bias
    layer22 = layer2';
    layer22 = [layer2Bias'; layer22];
    layer22 = layer22';

    inputsTest = inputsTest';
%     inputsTest = mapminmax(inputsTest');

    % try my own forward propagation with the trained wegiths.
    for j=1:1:row
       eachInput = [1 inputsTest(j,:)];
       eachInput = mapminmax(eachInput);
       for i=1:1:hiddenLayerSize
            L1_net(i) = eachInput*layer11(i,:)';
       end   
       L1_out = tansig(L1_net);
       L2_in = [1 L1_out];
       uptoClass = max(testLabel);
       for i=1:1:uptoClass
            L2_net(i) = L2_in*layer22(i,:)';
       end
       intSum = sum(exp(L2_net));
       L2_out(j,:) = exp(L2_net)/intSum;
    end

    L2_out = round(L2_out) ;

    class2 = vec2ind(L2_out');
    
    class2 = class2';
    
    err2 = errRate(class2, testLabel)
end
figure
plot(class, 'o')
title(['\fontsize{20}Train Result'])
xlabel(['\fontsize{20}Samples'])
ylabel(['\fontsize{20}Class'])
figure
plot(class2, 'o')
title(['\fontsize{20}Test Result'])
xlabel(['\fontsize{20}Samples'])
ylabel(['\fontsize{20}Class'])
% 
% if (1==0)
%     fid=fopen('layer1.txt','wt');
%     for i=1:1:hiddenLayerSize
%        for j=1:1:intputDim
%            fprintf(fid,'%.6f',layer11(i,j));
%            if (j==intputDim)
%              fprintf(fid,'\n');
%            else
%              fprintf(fid, ' ');
%            end
%        end
%     end
%     fclose(fid);
% 
%     fid=fopen('layer2.txt','wt');
%     for i=1:1:uptoClass
%        for j=1:1:hiddenLayerSize+1
%            fprintf(fid,'%.6f',layer22(i,j));
%            if (j==hiddenLayerSize+1)
%              fprintf(fid,'\n');
%            else
%              fprintf(fid, ' ');
%            end
%        end
%     end
%     fclose(fid);
% 
% 
%     [row, col] = size(testSet);
%     fid=fopen('testSet.txt','wt');
%     for i=1:1:row
%        for j=1:1:col
%            fprintf(fid,'%d',testSet(i,j));
%            if (j==col)
%              fprintf(fid,'\n');
%            else
%              fprintf(fid, ' ');
%            end
%        end
%     end
%     fclose(fid);
% 
%     [row, col] = size(testLabel);
%     fid=fopen('testLabel.txt','wt');
%     for i=1:1:row
%        for j=1:1:col
%            fprintf(fid,'%d',testLabel(i,j));
%            if (j==col)
%              fprintf(fid,'\n');
%            else
%              fprintf(fid, ' ');
%            end
%        end
%     end
%     fclose(fid);
% 
% end
