classdef CustomSiameseDatastore < matlab.io.Datastore
    % This creates the datastore for the siamese network.
    % Must be located within the same folder as the network file
    properties (SetAccess = private)
        Data % Cell array containing the data
        CurrentIndex % Current index for reading data
    end
    
    properties (SetAccess = private, Dependent)
        NumObservations % Number of observations in the custom datastore
    end
    
    methods
        function ds = CustomSiameseDatastore(data)
            ds.Data = data;
            ds.CurrentIndex = 1;
        end
        
        function numObservations = get.NumObservations(ds)
            numObservations = size(ds.Data, 1);
        end
        
        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end
        
        function reset(ds)
            % Reset the datastore to the beginning
            ds.CurrentIndex = 1;
        end
        
        function [data, info] = read(ds)
            % Read data from the custom datastore
            
            % Get the current index
            currentIndex = ds.CurrentIndex;
            
            % Read the data and labels at the current index
            rgbImage = single(ds.Data{currentIndex, 1});
            depthImage = single(ds.Data{currentIndex, 2});
            labels = ds.Data{currentIndex, 3};
            
            % Increment the index
            ds.CurrentIndex = ds.CurrentIndex + 1;
            
            % Set the output data and info structure
            data = {rgbImage, depthImage, labels};
            info.NumObservations = ds.NumObservations;
            info.CurrentIndex = currentIndex;
        end

        
        function frac = progress(ds)
            frac = ds.CurrentIndex / ds.NumObservations;
        end
    end
end
