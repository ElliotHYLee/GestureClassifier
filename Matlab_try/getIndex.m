function[result] = getIndex(data)
    label = zeros(1,length(data));
    for i=1:1:length(data)
        label(i) = getLabel(data(i,:));
    end
    result = label;
end

function[result] = getLabel(row)
    
    result = find(row==1);
    [a b] = size(result);
    if (b==0)
        result = -1;
    end
end