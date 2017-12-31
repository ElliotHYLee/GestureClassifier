clc, clear, close all;

L1w = importdata('layer1W.mat');
L2w = importdata('layer2W.mat');

testData = importdata('TestSet.mat');
% testData = testData(1:7940,:);


[row, col] = size(testData);
testSet = round(testData(:,1:col-1)/2*1000);
testLabel = testData(:,col);


targetsTest = zeros(row, 8);
for i=1:1:row
   temp = testLabel(i);
   targetsTest(i,temp) = 1;
end


for j=1:1:row
   eachInput = [1 testSet(j,:)];
   eachInput = mapminmax(eachInput);
   for i=1:1:10
        L1_net(i) = eachInput*L1w(i,:)';
   end   
   L1_out = tansig(L1_net);
   L2_in = [1 L1_out];
   uptoClass = max(testLabel);
   for i=1:1:uptoClass
        L2_net(i) = L2_in*L2w(i,:)';
   end
   intSum = sum(exp(L2_net));
   L2_out(j,:) = exp(L2_net)/intSum;
end

% L2_out = round(L2_out);

class = vec2ind(L2_out');
err = errRate(class, testLabel)
plot(class, 'o')


