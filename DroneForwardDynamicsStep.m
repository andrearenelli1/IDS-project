function [s_next] = DroneForwardDynamicsStep(s, u, m, dt)
% DroneMotionModel Simple point mass model of a drone
% No air resistance
% No rotor/motor dynamics
% No attitude coupling or thrust direction constraints 
% No orientation (roll/pitch/yaw)

%-----SANITY CHECKS-----
assert(isvector(s) && numel(s)==6, 'State must be 1x6');
assert(isvector(u) && numel(u)==3, 'Input must be 1x3');
assert(m > 0, 'Mass must be positive');
assert(dt > 0, 'dt must be positive');

% Gravity
g = [0 0 -9.81];

% Noise
sigma = 0.1;
w = sigma * randn(1,3);

% Acceleration
Acc = (u + w)/m + g;

% Velocity update
v_next = s(4:6) + Acc*dt;

% Position update
p_next = s(1:3) + s(4:6)*dt + 0.5*Acc*dt^2;

% Combine
s_next = [p_next, v_next];

end
