%{
"""Extract Histogram of Gradient features                           

Parameters                                                          
----------
X : ndarray NxHxW array where N is the number of instances/images   
                              HxW is the image dimensions           

Returns
    features : NxD narray contraining the histogram features (D = 144) 
-------                                                                                                                           
"""  
%}

function features = hog(X)

    % set up variables for HoG operation
    cell_size = [8, 8];
    [img_width,img_height] = size(X,2,3);
    num_bins = 9;
    G_mag = zeros(size(X));
    G_dir = zeros(size(X));
    features = zeros(size(X,1), img_width/cell_size(1) * img_height/cell_size(2) * num_bins);
   
    % compute gradient information for each image
    for i=1:size(X,1)
       I = squeeze(X(i,:,:));
       
       % apply a Sobel filter for each gradient (x,y) of kernel size 1
       [Gx, Gy] = imgradientxy(I);
       [magnitude, direction] = imgradient(Gx, Gy);
       
       % Calculate the mag and unsigned angle in range ( [0-180) degrees) for each pixel
       % mag, ang should have size NxHxW
       G_mag(i,:,:) = magnitude;
       G_dir(i,:,:) = direction;
    end  
    
    % Split orientation matrix/tensor into 8x8 cells
    G_dir = split_into_cells(G_dir, cell_size);
    
    % Split magnitude matrix/tensor into 8x8 cells, flattened to 64
    G_mag = split_into_cells(G_mag, cell_size);
    
    % compute HoG for all input images
    for v=1:size(X,1)
        % create an array to hold the feature histogram for each 8x8 cell in a image
        H = zeros(img_width/cell_size(1),img_height/cell_size(2), num_bins);
        m = squeeze(G_mag(v,:,:));
        d = squeeze(G_dir(v,:,:));
        
        % Loop through and for each cell calculate the histogram of gradients
        for i=1:size(H,1)
           for j=1: size(H,2)
               H(i,j,:) = calculate_hog_for_cell(m(i,j), d(i,j));
           end
        end
        
        %Normally, there is a window normalization step here, but we're going to ignore that.
        
        % Reshape the histogram so that its NxD where N is the number of instances/images i
        % and D is all the histograms per image concatenated into 1D vector
        features(v,:) = H(:);
        
    end
    
end

%{
    """Compute histogram of gradients for this a cell

    Parameters
    ----------
    C = Cell array of size 8x8

    Returns
    h = 1x9 histogram of gradients for this cell

    """
%}
function h = calculate_hog_for_cell(magnitudes_c, directions_c)
    
    h = zeros(1,9);
    magnitudes = cell2mat(magnitudes_c);
    directions = cell2mat(directions_c);
    
    for i=1:size(magnitudes_c,1)
        for j=1:size(magnitudes_c,2)
            r = mod(abs(directions(i,j)), 20);
            bin = ceil(abs(directions(i,j)/20));
            if(bin ==0)
                bin = 1;
            end
            if (r==0)
                % no remainder, so put the value direcetly into its
                % corresponding bin
                h(bin) = h(bin) + magnitudes(i,j);
            else
                % find closest bins 
                bin_next = 1 + mod(bin-1, 9);
                
                % calculate bin ratios
                h(bin) = h(bin) + (20-r)/20 * magnitudes(i,j);
                h(bin_next) = h(bin_next) + r/20 * magnitudes(i,j);
            end
            
        end
    end

end


%{
    """Split ndarray into smaller array

    Parameters
    ----------
    A : ndarray of size NxHxW
    cell : tuple with (h,w) for cell size 

    Returns
    -------
    ndarray of size Nx(cell_h*cell_w)x(cell_h*cell_w)
    
    """
%}
function B = split_into_cells(A, cell_size)

    % make sure cell_size input is of correct size and has square values
    if(~isempty(cell_size) || size(cell_size, 2) ~= 2 || cell_size(1) ~= cell_size(2))
       % our cell_array input is not good, output error 
    end
    
    % create vectors for cell2mat function
    h(1:size(A,2)/cell_size(1)) = cell_size(1);
    v(1:size(A,3)/cell_size(2)) = cell_size(2);
    
    % divide all input image into cell arrays
    for i=1:size(A,1)
        % grab the input image
        I = squeeze(A(i,:,:));
        
        % split the input image into cell arrays
        C = mat2cell(I, h, v);
        
        % store the new cell arrays into the output 
        B(i,:,:) = C;
        
    end
    
end
