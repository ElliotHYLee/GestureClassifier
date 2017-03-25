clc, clear, close all

type = 'Train';
numPpl = 3;
n = 2000*numPpl;
p{1} = 'chum';
p{2} = 'xu';
p{3} = 'lee';

%className = {'idle','takeoff','landing','moveright','moveleft','proceed','retreat','ccw','cw',};
className = {'idle', 'takeoff','moveright','moveleft','proceed','retreat','ccw','cw'};

for label=1:1:length(className)
   data_sub{label} =  getEachClass(className{label}, label, p, type, n);
end

for i=1:1:length(className)
    if i==1
        data = data_sub{i};
    else
        data = [data; data_sub{i}];
    end
end

data_mixed = mixData(data);

savetxt(type,data_mixed)

