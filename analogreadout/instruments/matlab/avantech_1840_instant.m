% avantech_1840_instant.m
%
% Matlab(2010 or 2010 above)
%
% Description:
%    This function acquires a data trace from the Advantech PCIE 1840 using
%    the instant routine. This is generally faster than
%    avantech_1840_acquire.m for nSamples < 10000, but the data is not
%    taken at a constant rate. 
%
% Args:
%   nChannels: integer
%       The number of channels to read out. The output data will be sliced
%       as data(channelNumber:nChannels:nSamples*nChannels). The channel
%       number is not a settable property, so to readout the 3rd channel,
%       for example, all the channels up to and including the 3rd must be
%       read (nChannels = 3).
%   nSamples: integer
%       The number of samples per channel to return when data is taken.
%
% Returns:
%   data: double
%       The measured data. Data from a particular channel can be returned
%       using 'data(channelNumber:nChannels:nSamples*nChannels)'.
%   errorStr: string
%       If no errors occured, an empty string is returned. If an error
%       occured, the error message is returned. This function should never
%       throw an error, and it is left up to the calling function to
%       perform error control
%
function [data, errorStr] = avantech_1840_instant(nChannels, nSamples)
% Make Automation.BDaq assembly visible to MATLAB.
BDaq = NET.addAssembly('Automation.BDaq4');
% Device info
deviceDescription = 'PCIE-1840,BID#0'; 
channelCount = int32(nChannels);
errorCode = Automation.BDaq.ErrorCode.Success;
errorStr = "";
data = zeros(1, nChannels * nSamples);

% Step 1: Create a 'InstantAiCtrl' for buffered AI function.
instantAiCtrl = Automation.BDaq.InstantAiCtrl();
try
    % Step 2: Select a device by device number or device description and
    % specify the access mode. in this example we use 
    % ModeWrite(default) mode so that we can 
    % fully control the device, including configuring, sampling, etc.
    instantAiCtrl.SelectedDevice = Automation.BDaq.DeviceInformation(...
            deviceDescription);
    % Step 3: Read samples
    currentData = NET.createArray('System.Double', channelCount);
    for ii = 1:nSamples
        errorCode = instantAiCtrl.Read(int32(0), channelCount, currentData);
        index = (channelCount * ii - channelCount + 1: channelCount * ii);
        data(1, index) = double(currentData);
        if avantech_1840_error(errorCode)
            throw Exception();
        end
    end
catch e
    % Something is wrong.
    preface = "An Advantech PCIe-1840 error occurred." + ...
        " And the last error code is:";
    if avantech_1840_error(errorCode)    
        errorStr = preface + " " + string(errorCode.ToString());
    else
        errorStr = preface + newline + string(e.message);
    end
end
% Step 4: Close device, release any allocated resource.
instantAiCtrl.Dispose();
end