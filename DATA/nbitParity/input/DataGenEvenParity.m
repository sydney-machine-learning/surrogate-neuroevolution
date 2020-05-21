% Even Parity data generator
bits = 4; % Input number of bites here
input = zeros(2^bits,bits);
counter = 0;
for i = 0: bits
    s = factorial(bits)/(factorial(i)*factorial(bits-i));
    tempin = zeros(s,bits);
    sampledata = [ones(1,i) zeros(1,bits-i)];
    for j = 1:s
        if s == 1
            if counter > 0
                tempin = ones(1,bits);
            end
            break;
        else
            cont = true;
            while cont
                temp = tempin(1:j-1,:);
                newdata = sampledata(randperm(bits));
                check = 0;
                for k = 1:j-1
                    rsum = sum(abs(temp(k,:) - newdata));
                    if rsum == 0
                        check = 1;
                        break;
                    end
                end
                if check == 0
                    tempin(j,:) = newdata;
                    cont = false;
                end
            end
        end
    end
    input(counter+1:counter+s,:) = tempin;
  
    data_parity = input
    counter = counter+s;
end
% input
dlmwrite('data4bits.txt',input)
