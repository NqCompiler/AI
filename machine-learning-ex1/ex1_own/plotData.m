function plotData( x, y )
%PLOTDATA uses the plot function to plot a two dimensional
% dataset, see help plot to learn more about this command
% X = independant variable
% y = target variable

plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

end

