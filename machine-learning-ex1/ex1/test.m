% Machine Learning : Linear Regression

clear all; close all; clc;

%% ======================= Plotting Training Data =======================
fprintf('Plotting Data ...\n')

data = load('ex1data1.txt');
x = data(:, 1); y = data(:, 2);
m = length(y); 

% Plot Data
plot(x,y,'rx');
xlabel('X -> Input') % x-axis label
ylabel('Y -> Output') % y-axis label

%% =================== Initialize Linear regression parameters ===================
 m = length(y); % number of training examples

% initialize fitting parameters - all zeros
theta=zeros(2,1);%theta 0,1

% Some gradient descent settings
iterations = 1500;
Learning_step_a = 1; % step parameter

%% =================== Gradient descent ===================

fprintf('Running Gradient Descent ...\n')

%Compute Gradient descent

% Initialize Objective Function History
J_history = zeros(iterations, 1);

m = length(y); % number of training examples

% run gradient descent    
for iter = 1:iterations

   % In every iteration calculate hypothesis
   hypothesis=theta(1)+theta(2).*x ;

   % Update theta variables
   temp0=theta(1) - Learning_step_a * (1/m) *sum(hypothesis-y);
   temp1=theta(2) - Learning_step_a * (1/m)* sum((hypothesis-y).* x);
  

   theta(1)=temp0;
   theta(2)=temp1;

   % Save objective function 
   J_history(iter)=(1/2*m)*sum(( hypothesis-y ).^2);

end

% print theta to screen
fprintf('Theta found by gradient descent: %f %f\n',theta(1),  theta(2));
fprintf('Minimum of objective function is %f \n',J_history(iterations));

% Plot the linear fit
hold on; % keep previous plot visible 
plot(x, theta(1)*x+theta(2), '-')

% Validate with polyfit fnc
poly_theta = polyfit(x,y,1);
plot(x, poly_theta(1)*x+poly_theta(2), 'y--');
legend('Training data', 'Linear regression','Linear regression with polyfit')
hold off 

figure
% Plot Data
plot(x,y,'rx');
xlabel('X -> Input') % x-axis label
ylabel('Y -> Output') % y-axis label

hold on; % keep previous plot visible
% Validate with polyfit fnc
poly_theta = polyfit(x,y,1);
plot(x, poly_theta(1)*x+poly_theta(2), 'y--');

% for theta values that you are saying
theta(1)=0.0745;  theta(2)=0.3800;
plot(x, theta(1)*x+theta(2), 'g--')
legend('Training data', 'Linear regression with polyfit','Your thetas')
hold off 