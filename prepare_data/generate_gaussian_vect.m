function [ data ] = generate_gaussian_vect( mu, sigma, minval, maxval, num)

num_invalid = 1;
while num_invalid
    data = normrnd(mu, sigma, num, 1);
    num_invalid = length(find((data < minval) | (data > maxval)));
end

end

