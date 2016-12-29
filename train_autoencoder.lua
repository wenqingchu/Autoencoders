-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end

-- Load MNIST data
local XTrain = mnist.traindataset().data:float():div(255) -- Normalise to [0, 1]
local XTest = mnist.testdataset().data:float():div(255)
local N = XTrain:size(1)
if cuda then
  XTrain = XTrain:cuda()
  XTest = XTest:cuda()
end

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-cpu', true, 'CPU only (useful if GPU memory is too low)')
cmd:option('-model', 'AE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|CatVAE|AAE|WTA-AE')
cmd:option('-learningRate', 0.01, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 100, 'Training epochs')
cmd:option('-mcmc', 0, 'MCMC samples')
cmd:option('-sampleStd', 1, 'Standard deviation of Gaussian distribution to sample from')
local opt = cmd:parse(arg)
opt.batchSize = 150 -- Currently only set up for divisors of N
if opt.cpu then
  cuda = false
end

-- Create model
local Model = require ('models/' .. opt.model)
Model:createAutoencoder(XTrain)
local autoencoder = Model.autoencoder
if cuda then
  autoencoder:cuda()
  -- Use cuDNN if available
  if hasCudnn then
    cudnn.convert(autoencoder, cudnn)
  end
end

-- Create adversary (if needed)
local adversary
if opt.model == 'AAE' then
  Model:createAdversary()
  adversary = Model.adversary
  if cuda then
    adversary:cuda()
    -- Use cuDNN if available
    if hasCudnn then
      cudnn.convert(adversary, cudnn)
    end
  end
end

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()
local thetaAdv, gradThetaAdv
if opt.model == 'AAE' then
  thetaAdv, gradThetaAdv = adversary:getParameters()
end

-- Create loss
local criterion = nn.BCECriterion()
--local criterion = nn.MSECriterion()
local softmax = nn.SoftMax() -- Softmax for CatVAE KL divergence
if cuda then
  criterion:cuda()
  softmax:cuda()
end

-- Create optimiser function evaluation
local x -- Minibatch
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()
  if opt.model == 'AAE' then
    gradThetaAdv:zero()
  end

  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  -- Regularization phase
  if opt.model == 'Seq2SeqAE' then
    -- Clamp RNN gradients to prevent exploding gradients
    gradTheta:clamp(-10, 10)
  elseif opt.model == 'VAE' then
    -- Optimise Gaussian KL divergence between inference model and prior: DKL[q(z|x)||N(0, σI)] = log(σ2/σ1) + ((σ1^2 - σ2^2) + (μ1 - μ2)^2) / 2σ2^2
    local nElements = xHat:nElement()
    local mean, logVar = table.unpack(Model.encoder.output)
    local var = torch.exp(logVar)
    local KLLoss = 0.5 * torch.sum(torch.pow(mean, 2) + var - logVar - 1)
    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
    loss = loss + KLLoss
    local gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}  -- Normalise gradient of loss (same normalisation as BCECriterion)
    Model.encoder:backward(x, gradKLLoss)
  elseif opt.model == 'CatVAE' then
    -- Optimise KL divergence between inference model and prior
    local nElements = xHat:nElement()
    local z = softmax:forward(Model.encoder.output:view(-1, Model.k)) + 1e-9 -- Improve numerical stability
    local logZ = torch.log(z)
    local KLLoss = torch.sum(z:cmul(logZ - math.log(1 / Model.k)))
    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
    local gradKLLoss = softmax:backward(Model.encoder.output:view(-1, Model.k), math.log(1 / Model.k) - logZ - 1):view(-1, Model.N * Model.k)
    gradKLLoss = gradKLLoss / nElements -- Normalise gradient of loss (same normalisation as BCECriterion)
    loss = loss + KLLoss
    Model.encoder:backward(x, gradKLLoss)
    
    -- Anneal temperature τ
    Model.tau = math.max(Model.tau - 0.0002, 0.5)
    Model.temperature.constant_scalar = 1 / Model.tau
  elseif opt.model == 'AAE' then
    local real = torch.Tensor(opt.batchSize, Model.zSize):normal(0, 1):typeAs(XTrain) -- Real samples ~ N(0, 1)
    local YReal = torch.ones(opt.batchSize):typeAs(XTrain) -- Labels for real samples
    local YFake = torch.zeros(opt.batchSize):typeAs(XTrain) -- Labels for generated samples

    -- Train adversary to maximise log probability of real samples: max_D log(D(x))
    local pred = adversary:forward(real)
    local realLoss = criterion:forward(pred, YReal)
    local gradRealLoss = criterion:backward(pred, YReal)
    adversary:backward(real, gradRealLoss)

    -- Train adversary to minimise log probability of fake samples: max_D log(1 - D(G(x)))
    pred = adversary:forward(Model.encoder.output)
    local fakeLoss = criterion:forward(pred, YFake)
    advLoss = realLoss + fakeLoss
    local gradFakeLoss = criterion:backward(pred, YFake)
    local gradFake = adversary:backward(Model.encoder.output, gradFakeLoss)

    -- Train encoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
    local minimaxLoss = criterion:forward(pred, YReal) -- Technically use max_G max_D log(D(G(x))) for same fixed point, stronger initial gradients
    loss = loss + minimaxLoss
    local gradMinimaxLoss = criterion:backward(pred, YReal)
    local gradMinimax = adversary:updateGradInput(Model.encoder.output, gradMinimaxLoss) -- Do not calculate gradient wrt adversary parameters
    Model.encoder:backward(x, gradMinimax)
  end

  return loss, gradTheta
end

local advFeval = function(params)
  if thetaAdv ~= params then
    thetaAdv:copy(params)
  end

  return advLoss, gradThetaAdv
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate = opt.learningRate}
local advOptimParams = {learningRate = opt.learningRate}
local __, loss
local losses, advLosses = {}, {}


for epoch = 1, opt.epochs do
  local  trainloss = 0
  
  for n = 1, N, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)

    -- Optimise
    __, loss = optim[opt.optimiser](feval, theta, optimParams)
    losses[#losses + 1] = loss[1]
    trainloss = trainloss + loss[1]

    -- Train adversary
    if opt.model == 'AAE' then
      __, loss = optim[opt.optimiser](advFeval, thetaAdv, advOptimParams)     
      advLosses[#advLosses + 1] = loss[1]
    end
  end
  print('Epoch ' .. epoch .. '/' .. opt.epochs .. ' loss: ' .. trainloss/N*opt.batchSize)

end



-- Test
print('Testing for traindata')
local xHat
local featuresize = 2000
feat = torch.Tensor(N, featuresize)
for n = 1, N, 100 do
    -- Get minibatch
    x = XTrain:narrow(1, n, 100)
    autoencoder:evaluate()
    xHat = autoencoder:forward(x)
    --conv_nodes = autoencoder:findModules('nn.ReLU')
    feat[{{n,n+100-1}, {1,featuresize}}] = Model.encoder.output
end


file = torch.DiskFile('WTAAE_train.asc', 'w')
file:writeObject(feat)
file:close()



-- Test
print('Testing')
local TN = XTest:size(1)
feat = torch.Tensor(TN, featuresize)
for n = 1, TN, 100 do
    -- Get minibatch
    x = XTest:narrow(1, n, 100)
    autoencoder:evaluate()
    xHat = autoencoder:forward(x)
    --conv_nodes = autoencoder:findModules('nn.ReLU')
    feat[{{n,n+100-1}, {1,featuresize}}] = Model.encoder.output
end


file = torch.DiskFile('WTAAE_test.asc', 'w')
file:writeObject(feat)
file:close()
--if opt.model == 'AE' then
--  local feat = autoencoder.modules[3].output
--  print(feat)
--end

