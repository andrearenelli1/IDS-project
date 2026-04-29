function [s_next] = DroneForwardDynamicsStep(s, u, m, dt, v_wind)
% DroneMotionModel Simple point mass model of a drone
% No air resistance
% No rotor/motor dynamics
% No attitude coupling or thrust direction constraints 
% No orientation (roll/pitch/yaw)

%-----SANITY CHECKS-----
assert(isvector(s) && numel(s)==6, 'State must be 1x6');
assert(isvector(u) && numel(u)==3, 'Input must be 1x3');
assert(isvector(v_wind) && numel(v_wind)==3, 'Wind speed vector must be 1x3');
assert(m > 0, 'Mass must be positive');
assert(dt > 0, 'dt must be positive');

% IN State
p = s(1:3);
v = s(4:6);

% Gravity
g = [0 0 -9.81];

% Noise on actuation
sigma = 0.1;
w = sigma * randn(1,3);

% Wind effect
v_rel = v - v_wind; 
if norm(v_rel) <= 0
    v_rel = [0 0 0];
end

% Quadratic Drag
k_xy = 0.05;
k_z  = 0.1;
F_drag = -norm(v_rel)*[k_xy*v_rel(1), k_xy*v_rel(2), k_z*v_rel(3)];

% Acceleration
Acc = (u + w + F_drag)/m + g;

% Velocity update
v_next = v + Acc*dt;

% Position update
p_next = p + v*dt + 0.5*Acc*dt^2;

% Combine
s_next = [p_next, v_next];

end
