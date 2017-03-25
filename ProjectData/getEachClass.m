function[result] = getEachClass(class, label, p, type,n)
    for i=1:1:length(p)
        temp{i} = getData(p{i}, class, type);
    end
    result = [temp{1};temp{2};temp{3}];
    result = [label*ones(1,n)' result];
end


function[d1] = getData(p, class, type)
    d1 = importdata(strcat('RawData\', p, '_', class, type, '.txt'));
end