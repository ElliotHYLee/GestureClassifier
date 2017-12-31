function [result] = errRate(y, y_pred)
    for i=1:1:length(y)
        if(y(i)~=y_pred(i))
            err(i) = 1;
        else
            err(i) = 0;
        end
    end
    
    number = nnz(err);
    
    result = number/length(y)*100;
    
end