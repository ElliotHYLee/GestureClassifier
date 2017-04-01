function [result] = errRate(targetsTest, outputs)
    errCnt = 0;
    for i = 1:1:length(outputs)
       est_y = outputs(i,:);
       des_y = targetsTest(i,:);
       if(isequal(est_y,des_y))
       else
           errCnt = errCnt + 1;
       end
    end
    
    result = errCnt/length(outputs);
end