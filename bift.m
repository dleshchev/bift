function [rgrid, P, Perr, Ifit, Rg, I0 ] = bift( q, I, I_err, varargin )
%BIFT Calculates the inverse Fourier Transform using Bayes inference (v0.0)
%   This function utilizes method outlined in S. Hansen, J. Appl. Cryst.
%   (2000) 33, 1415-1421. The probaility notation is taken from A.H. Larsen
%   J. Appl. Cryst. (2018) 51, 1151–1161, equation 5.
%   
%   Input parameters:
%   q - momentum transfer values (in A^-1, should be spaced equidistanly).
%
%   I - measured intensity
%
%   I_err - uncertainties (1-sigma) of measured intensities
%
%   Nr - parameter controlling the density of output rgrid points. Default
%   value is 52. The more, the denser is rgrid. As a rule of thumb, Nr
%   will be approximately equal to the number of P(r) points that are not
%   equal to 0 within uncertainty Perr(r). If you want to leave it as is
%   and supply other input parameters, then set this argument to [].
%
%   plotting - flag to control if the routine will plot the results.
%   Default is true.
%
%   initGrid - the parameters of the initial grid of parameters. It is
%   defined as follows:
%   initGrid = [logLambdaMin, logLambdaMax, DmaxMin, DmaxMax, npt]
%   logLambdaMin - log of minimum lambda value in the initial grid search.
%   Default value is 0.
%   logLambdaMax - log of maximum lambda value in the initial grid search.
%   Default value is 10.
%   DmaxMin - minimum Dmax value (in angstrom) in the initial grid search.
%   Default value is 2 Angstrom.
%   DmaxMax - maximum Dmax value in the initial grid search. Default value
%   is 500 Angstrom.
%   npt - number of points along each axis to perform initial grid
%   search. Default value is 31.
%
%   postDensity - density of points per one standard deviation of lambda
%   and Dmax in the final posterior calculation. Default value is 3. More
%   points take more time for calculation.
%
%   Output parameters:
%   rgrid - the final grid of r values 
%   P - calculated pair distibution function
%   Perr - uncertainties in pair distribution function
%   Ifit - fitted intensity
%   Rg - estimated radius of gyration
%   I0 - estimated I0
%
%   METHOD DESCRIPTION:
%   The approach combines the prior knowledge about the  pair distribution
%   function (smoothness) and the experimental data. 
%   In this approach, the intensity is expressed as FT of pair distribution
%   function:
%   I(q) = 4 * pi * integral(0, Dmax) { P(r) sin(q * r) / (q * r) dr }
%   Or, equivently, in a matrix form:
%   I(q) = T * P(r)
%   where T_{i,j} = 4 * pi * sin(q_i * r_j) / (q_i * r_j) * dr
%   Direct solution of this problem can be obtained via minimizing
%   chisq =  sum(q) {((I(q) - T * P(r))./I_err).^2}
%   The solution to this problem is produces highly oscillatory solutions,
%   so a penalty term is introduced:
%   S = sum(i=2:Nr-1) {(P_i - (P_{i-1} + P_{i+1})/2)^2} + 0.5*P_i^2 +
%   0.5*P_Nr^2
%   where Nr is the number of r-points. The first term in this expression
%   penalizes non-smooth solutions, whereas the second and the third terms
%   penalize the first and the last points in P(r), forcing P(0) and
%   P(Dmax) to be zero.
%   With the penalty, the solution is found via minimization of the
%   following cost function:
%   Q = chisq + lambda * S
%   where lambda controls the amount of penalty on the solution.
%   If W = diag(1./I_err.^2) and L is defined such that S = P' * L' * L * P
%   then minimum of Q can be found as
%   P(r, lambda, Dmax) = (T' * W * T + lambda * L' * L)^-1 * T' * W * I
%   There is a unique solution for each pair of (lambda, Dmax). Therefore,
%   it is necessary to find appropriate pair of these hyper parameters.
%   In the Hansen approach, such pair is found via calculation of the
%   posterior probability. If POSTERIOR is the posterior probability, it
%   can be expressed as follows (A.H. Larsen, J. Appl. Cryst. (2018). 51,
%   1151–1161):
%   -2*log(POSTERIOR) = Q + log(G) + 2*log(lambda)
%   where G = det(Hess(Q))/det(Hess(lambda * S)), and Hess is the Hessian
%   operator. Simple math shows that 
%   Hess(Q) = T' * W * T + lambda * (L' * L)
%   Hess(lambda * S) = lambda * (L' * L)
%   Maximum of POSTERIOR shows the most likely coombination of lambda and
%   Dmax. To find final P, one has to get a weighted sum of different
%   probabilities:
%   <P(r)> = sum_(lambda, Dmax) P(r, lambda, Dmax) * POSTERIOR(lambda, Dmax)
%   One can also estimate the variance  in P(r):
%   Var(P(r)) = <P(r).^2> - <P(r)>.^2
%
%   Practically, we implement this approach as follows:
%   1. Coarse search on a grid between lambdaMin, lambdaMax, DmaxMin,
%   DmaxMax.
%   2. Starting from the maximum Posterior estimate from the coarse grid
%   search, we find for the maximum using convex optimization (fminunc)
%   3. Final calculation of posterior around the optimum point
%   4. Calculation of the <P(r)> and Var(P(r)) using estimated POSTERIOR
%
%   Denis Leshchev, Chen Lab, Northwestern University (Feb 2019)
%   
%% Input control
numvarargs = length(varargin);
if numvarargs > 4
    error('myfuns:bift:TooManyInputs', ...
          'requires at most 4 optional inputs');
end

% default values
optargs = {52, true, [0, 10, 2, 500, 31], 3};

% update default vaues if they are provided
for i = 1:numvarargs
    if ~isempty(varargin{i})
        optargs{i} = varargin{i};
    end
end
[Nr, plotting, initGrid, postDensity] = optargs{:};

%% Data prep
% Reshape the data
Nq = length(q); % number of q points
q = reshape(q, Nq, 1);
I = reshape(I, Nq, 1);
I_err = reshape(I_err, Nq, 1);

% Renormalize the data
normFactor = norm(I);
I = I/normFactor;
I_err = I_err/normFactor;

% Initial grid parameters
logLambdaMin = initGrid(1);
logLambdaMax = initGrid(2);
DmaxMin = initGrid(3);
DmaxMax = initGrid(4);
npt_coarse = initGrid(5);

%% Search for a good initial start
lambda_init = 10.^linspace(logLambdaMin, logLambdaMax, npt_coarse)';
Dmax_init = linspace(DmaxMin, DmaxMax, npt_coarse)';

doubleNegLogPosterior_init = zeros(length(lambda_init), length(Dmax_init));
for i = 1:length(lambda_init)
    for j = 1:length(Dmax_init)
        doubleNegLogPosterior_init(i, j) = getSolution(I, I_err, lambda_init(i), Dmax_init(j));
    end
end

[~, idx] = min(doubleNegLogPosterior_init(:));
[idx_row, idx_col] = ind2sub(size(doubleNegLogPosterior_init), idx);
par_start = [log(lambda_init(idx_row)), Dmax_init(idx_col)];

%% Search for the optimum
doubleNegLogPosterior_fun = @(a) getSolution(I, I_err, exp(a(1)), a(2));
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton', 'Display', 'Off');
[par_opt, doubleNegLogPosterior_opt, ~, ~, ~, Hess] = fminunc(doubleNegLogPosterior_fun, par_start, options);

logLambda_opt = par_opt(1);
Dmax_opt = par_opt(2);

par_std = sqrt(diag(pinv(Hess)));
logLambda_std = par_std(1);
Dmax_std = par_std(2);

%% Final grid calculation
% To define the final grid we need to find the meaningful range in (lambda,
% Dmax) plane where posterior will be more than 0. To do so, we will find
% the ranges where doubleNegLogPosterior increases by 25, which resembles
% 5-sigma rule.
[lambda_grid, Dmax_grid] = getFinalGrid(25, postDensity);

% final rgrid is defined using Dmax_opt and extended up to Dmax_grid with
% appropriate step size:
rgrid = getRgrid(Dmax_opt);
rgrid = [ rgrid(1:end-1); ...
         (rgrid(end) : (rgrid(2)-rgrid(1)) : max(Dmax_grid))'];

% value initialization for the final posterior probability calculation:
P = zeros(size(rgrid));
Perr = zeros(size(rgrid));
Nind = 0;

doubleNegLogPosterior_grid = zeros(length(lambda_grid), length(Dmax_grid));
posterior = zeros(length(lambda_grid), length(Dmax_grid));

% commented parts of the loop below are for lazy integration. Its usage is
% descouraged
for j = 2:length(Dmax_grid)-1
    for i = 2:length(lambda_grid)-1
        dlambda = (lambda_grid(i+1) - lambda_grid(i-1))/2;
        dDmax = (Dmax_grid(j+1) -  Dmax_grid(j-1))/2;
% for j = 1:length(Dmax_grid)
%     for i = 1:length(lambda_grid)
        [doubleNegLogPosterior_grid(i, j), r_loc, P_loc, Nind_loc] = ...
            getSolution(I, I_err, lambda_grid(i), Dmax_grid(j));
        posterior(i, j) = exp(-0.5*(doubleNegLogPosterior_grid(i, j) - doubleNegLogPosterior_opt))*dlambda*dDmax;
%         posterior(i, j) = exp(-0.5*(doubleNegLogPosterior_grid(i, j) - doubleNegLogPosterior_opt));
        P_loc_interp = interp1(r_loc, P_loc, rgrid, 'pchip', 0);
        P = P + posterior(i, j)*P_loc_interp;
        Perr = Perr + posterior(i, j)*P_loc_interp.^2;
        Nind = Nind + posterior(i, j)*Nind_loc;
    end
end

%% Renormalization and output calculation
normScale = sum(posterior(:));
posterior = posterior/normScale;
P = P/normScale;
Perr = sqrt(Perr/normScale - P.^2)*normFactor;
P = P*normFactor;
Nind = Nind/normScale;
Tfinal = getTmatrix(rgrid);
Ifit = Tfinal*P;
I = I*normFactor;
I_err = I_err*normFactor;
chisqOpt = (norm((Ifit - I)./I_err)^2);
chisqOptRed = chisqOpt/(Nq - Nind - 1);

Rg = sqrt(trapz(rgrid, rgrid.^2.*P)/trapz(rgrid, P)/2);
I0 = 4*pi*trapz(rgrid, P);

%% Plotting
if plotting
    figure();
    clf();
    subplot(221)
    mesh(Dmax_grid, lambda_grid, posterior);
    hold on
    set(gca,'yscale','log')
    xlabel(['Dmax, ', char(197)])
    ylabel('\lambda')
    zlabel('Posterior Probability')

    subplot(222); hold on;
    errorbar(rgrid, P, Perr, 'k.-')
    xlabel(['r, ', char(197)])
    ylabel('P(r)')
    legend(sprintf( ['Rg = ', num2str(round(Rg, 2)), '\n', ...
                     'I0 = ', num2str(round(I0, 3))]))
    
    subplot(223); hold on;
    plot(q, I, 'k.-')
    plot(q, Ifit, 'r-')
    legend('data', sprintf( ...
                    ['chi2 = ', num2str(round(chisqOpt,2)), '; ', ...
                     'chi2red = ', num2str(round(chisqOptRed,2)), '\n', ...
                     'Nq = ', num2str(round(Nq, 0)), '; ', ...
                     'N_{ind} = ', num2str(round(Nind, 1)), '\n', ...
                    ]))
    xlabel(['q, ', char(197), '^{-1}'])
    ylabel('I(q)')
end

%% AUX functions
function [r, T, L] = getMatrices(DmaxVal)
    r = getRgrid(DmaxVal);
    T = getTmatrix(r);
    L = getLmatrix();
end


function r = getRgrid(DmaxVal)
    r = linspace(1e-6, DmaxVal, Nr)';
end


function T = getTmatrix(r)
    dr = r(2)-r(1);
    T = 4*pi*sin(q.*r')./(q.*r')*dr;
end


function L = getLmatrix()
    L = zeros(Nr);
    for ii = 2:(Nr-1)
       L(ii, ii-1) = -1/2;
       L(ii, ii) = 1;
       L(ii, ii+1) = -1/2;
    end
    L(1,1) = 1/sqrt(2);
    L(end, end) = 1/sqrt(2);
end


function [doubleNegLogPosterior, r, P, Nind] = getSolution(I, I_err, lambdaVal, DmaxVal)
    [r, T, L] = getMatrices(DmaxVal);
    W = diag(1./I_err.^2);
    A = lambdaVal*(L'*L);
    B = T'*W*T;
    C = A + B;
    d = T'*W*I;
    P = pinv(C)*d;
    chisq = norm((T*P - I)./I_err)^2;
    penalty = lambdaVal*norm(L*P)^2;
    Aeig = eig(A);
    Beig = real(eig(B));
    Ceig = eig(C);
    Nind = sum(Beig./Ceig);
    logdetA = sum(log(Aeig));
    logdetC = sum(log(Ceig));
    logG = logdetC - logdetA;
    doubleNegLogPosterior = chisq + penalty + logG + 2*log(lambdaVal);
%     doubleNegLogPosterior = chisq + penalty + logG;
end


function [lambda_grid, Dmax_grid] = getFinalGrid(dValThresh, pointDensity)
    nLambdaPos = searchLambda(dValThresh, +1);
    nLambdaNeg = searchLambda(dValThresh, -1);
    nDmaxPos = searchDmax(dValThresh, +1);
    nDmaxNeg = searchDmax(dValThresh, -1);
    
    lambda_grid = linspace(nLambdaNeg, nLambdaPos, (-nLambdaNeg + nLambdaPos)*pointDensity + 1);
    lambda_grid = exp(logLambda_opt + lambda_grid' * logLambda_std);
    Dmax_grid = linspace(nDmaxNeg, nDmaxPos, (-nDmaxNeg + nDmaxPos)*pointDensity + 1);
    Dmax_grid = Dmax_opt + Dmax_grid' * Dmax_std;
end


function n = searchLambda(dValThresh, direction)
    dVal = 0;
    n = 0;
    while dVal < dValThresh
        n = n + direction;
        dVal = doubleNegLogPosterior_fun( ...
            [logLambda_opt + n * logLambda_std, Dmax_opt] ...
                                          ) - doubleNegLogPosterior_opt;
    end
end


function n = searchDmax(dValThresh, direction)
    dVal = 0;
    n = 0;
    while dVal < dValThresh
        n = n + direction;
        dVal = doubleNegLogPosterior_fun( ...
            [logLambda_opt, Dmax_opt + n * Dmax_std] ...
                                          ) - doubleNegLogPosterior_opt;
    end
end


end

