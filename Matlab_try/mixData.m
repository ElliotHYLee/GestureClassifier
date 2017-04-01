function[result] = mixData(data)
M = data;
r = randperm(size(M,1)); % permute row numbers
result = M(r,:);
end