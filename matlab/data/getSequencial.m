function[data1] = getSequencial(rawData, sequenceNumber)

    [row, col] = size(rawData);
    data1 = zeros((row-sequenceNumber), col*sequenceNumber);
    for i=1:1: (row-sequenceNumber)
        for j=1:1:sequenceNumber
            concatIndex = col*j-5;
            data1(i,concatIndex:concatIndex+5) = rawData(j+i,:);
        end    
    end


end