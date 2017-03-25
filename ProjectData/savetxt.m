function[] = savetxt(type, data)
    string = '';
    [r, c] = size(data);
    for i=1:1:c
        string = strcat(string, ' %d');
    end
    string  = strcat(string, '\n');


    fileID = fopen(strcat(type, '.txt'),'w');
    for i=1:1:length(data)
        fprintf(fileID, string ,data(i,:));
    end

    fclose(fileID);
end