function u = DronePIDControllerStep(s, s_ref, kp, kd)
% DronePIDController
% Computes control force for the point-mass drone model
%
% INPUTS:
% s      : current state [1x6] = [pos vel]
% p_ref  : desired position [1x3]
% v_ref  : desired velocity [1x3] (use NaN if not available)
% kp     : proportional gain (scalar or 1x3)
% kd     : derivative gain (scalar or 1x3)
%
% OUTPUT:
% u      : control force [1x3]

% Extract state
p = s(1:3);
v = s(4:6);

% Ensure row vectors
p_ref = s_ref(1:3)';
v_ref = s_ref(4:6)';

% Gains (allow scalar or vector)
if isscalar(kp), kp = kp * ones(1,3); end
if isscalar(kd), kd = kd * ones(1,3); end

% Position error
e_p = p_ref - p;

% Velocity error (handle missing components)
e_v = zeros(1,3);
e_v = v_ref - v;

% for i = 1:3
%     if ~isnan(v_ref(i))
%         e_v(i) = v_ref(i) - v(i);
%     else
%         % No derivative term on this axis
%         e_v(i) = 0;
%     end
% end

% Control Law
u = kp.*e_p + kd.*e_v;

% Saturation

% Anti-Windup

end