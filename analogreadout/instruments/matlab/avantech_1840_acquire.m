% avantech_1840_acquire.m
%
% Matlab(2010 or 2010 above)
%
% Description:
%    This function acquires a data trace from the Advantech PCIE 1840. The
%    digitizer must first be setup by avantech_1840_startup.m
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
function [data, errorStr] = avantech_1840_acquire()
% Make Automation.BDaq assembly visible to MATLAB.
BDaq = NET.addAssembly('Automation.BDaq4');
% Device info
deviceDescription = 'PCIE-1840,BID#0'; 
% Step 1: Create a 'WaveformAiCtrl' for buffered AI function.
waveformAiCtrl = Automation.BDaq.WaveformAiCtrl();
errorCode = Automation.BDaq.ErrorCode.Success;
errorStr = "";
data = [];

try
    % Step 2: Select a device by device number or device description and
    % specify the access mode. in this example we use 
    % ModeWrite(default) mode so that we can 
    % fully control the device, including configuring, sampling, etc.
    waveformAiCtrl.SelectedDevice = Automation.BDaq.DeviceInformation(...
        deviceDescription);
    % disp('SynchronousOneBufferedAI is in progress.');
    % disp('Please wait, until acquisition complete.');
    errorCode = waveformAiCtrl.Start();
    if avantech_1840_error(errorCode)
        throw Exception();
    end
    conversion = waveformAiCtrl.Conversion;
    record = waveformAiCtrl.Record;
    
    % Step 3: Read samples
    count = record.SectionLength * record.SectionCount * ...
        conversion.ChannelCount;
    data = NET.createArray('System.Double', count);
    %-1 means waiting for filling up the buffer.
    errorCode = waveformAiCtrl.GetData(count, data, -1);
    data = double(data);
    % disp('Acquisition has completed!');
    if avantech_1840_error(errorCode)
        throw Exception();
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
waveformAiCtrl.Dispose();
end

