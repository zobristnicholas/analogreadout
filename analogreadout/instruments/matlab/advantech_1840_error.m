% advantech_1840_error.m
%
% Matlab(2010 or 2010 above)
%
% Description:
%    This function determines if an Advantech error code is an error
%
% Returns:
%   result: boolean
%       True if errorCode corresponds to an error, false otherwise
%
function result = advantech_1840_error(errorCode)

result =  errorCode < Automation.BDaq.ErrorCode.Success && ...
    errorCode >= Automation.BDaq.ErrorCode.ErrorHandleNotValid;

end