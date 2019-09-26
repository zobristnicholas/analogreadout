% advantech_1840_startup.m
%
% Matlab(2010 or 2010 above)
%
% Description:
%    This function sets up the Advantech PCIE 1840 for data taking with
%    advantech_1840_acquire.m.
%
% Args:
%   nChannels: integer
%       The number of channels to read out. The output data will be sliced
%       as data(channelNumber:nChannels:nSamples*nChannels). The channel
%       number is not a settable property, so to readout the 3rd channel,
%       for example, all the channels up to and including the 3rd must be
%       read (nChannels = 3).
%   sampleRate: double
%       The sample rate in Hz of the digitizer. This code assumes that the
%       external reference clock is connected. Note that the set frequency
%       may differ slightly from the requested frequency.
%   nSamples: integer
%       The number of samples per channel to return when data is taken.
%       Note that the set number of samples may differ slightly from the
%       requested value.
%
% Returns:
%   sampleRate: double
%       The sample rate that was actually set on the digitizer.
%   nSamples: integer
%       The number of samples per channel that was actually set on the
%       digitizer.
%   errorStr: string
%       If no errors occured, an empty string is returned. If an error
%       occured, the error message is returned. This function should never
%       throw an error, and it is left up to the calling function to
%       perform error control
%
function [sampleRate, nSamples, errorStr] = advantech_1840_startup(...
    nChannels, sampleRate, nSamples)
% Make Automation.BDaq assembly visible to MATLAB.
BDaq = NET.addAssembly('Automation.BDaq4');

% Device info
deviceDescription = 'PCIE-1840,BID#0'; 
channelCount = int32(nChannels);
convertClkRate = int32(sampleRate);
sectionLength = int32(nSamples);
sectionCount = int32(1); %1 meas one buffered mode.

clockSource = Automation.BDaq.SignalDrop.SigExtDigRefClock;
errorCode = Automation.BDaq.ErrorCode.Success;
errorStr = "";
% Step 1: Create a 'WaveformAiCtrl' for buffered AI function.
waveformAiCtrl = Automation.BDaq.WaveformAiCtrl();

try
    % Step 2: Select a device by device number or device description and
    % specify the access mode. in this example we use 
    % ModeWrite(default) mode so that we can 
    % fully control the device, including configuring, sampling, etc.
    waveformAiCtrl.SelectedDevice = Automation.BDaq.DeviceInformation(...
        deviceDescription);

    % Step 3: Set necessary parameters for Asynchronous One Buffered AI
    % operation.
    valueRange = Automation.BDaq.ValueRange.V_Neg2To2;  % V_Neg5To5; % V_Neg1To1;  % mV_Neg100To100;
    for ii = 1:nChannels
        % fix at lowest voltage range
        waveformAiCtrl.Channels(ii).ValueRange = valueRange;
    end
    conversion = waveformAiCtrl.Conversion;
    record = waveformAiCtrl.Record;
    
    conversion.ChannelCount = channelCount;
    conversion.ClockRate = convertClkRate;
    % the set sample rate might be slightly different
    sampleRate = conversion.ClockRate; 
    conversion.ClockSource = clockSource;

    record.SectionCount = sectionCount;
    record.SectionLength = sectionLength;
    % the set number of samples might be slightly different
    if nSamples ~= 0
        nSamples = ceil(nSamples / 4) * 4;
    else
        nSamples = 1024;
    end
    % Step 4: Prepare the buffered AI. 
    errorCode =  waveformAiCtrl.Prepare();
    if advantech_1840_error(errorCode)
        throw Exception();
    end
catch e
    % Something is wrong.
    preface = "An Advantech PCIe-1840 error occurred." + ...
        " And the last error code is:";
    if advantech_1840_error(errorCode)    
        errorStr = preface + " " + string(errorCode.ToString());
    else
        errorStr = preface + newline + string(e.message);
    end
end

% Step 7: Close device, release any allocated resource.
waveformAiCtrl.Dispose();
end
