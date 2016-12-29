require 'nn'
require 'optim'


local mnist = require 'mnist'
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



-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-cpu', false, 'CPU only (useful if GPU memory is too low)')
cmd:option('-model', 'linear', 'Model: linear|mlp|convnet')
cmd:option('-learningrate', 0.1, 'Learning rate')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-batchsize', 20, 'batchsize')

local opt = cmd:parse(arg)

if opt.cpu then
  cuda = false
end


local trainset = mnist.traindataset()
local testset = mnist.testdataset()


batchsize = opt.batchsize

dataset_inputs = torch.Tensor(60000, 1, 28, 28)
dataset_outputs = torch.Tensor(60000)


for i=1,trainset.size do
    dataset_outputs[i] = trainset[i].y + 1
    dataset_inputs[i] = trainset[i].x
end

mean = dataset_inputs:mean()
std = dataset_inputs:std()

dataset_inputs:add(-mean)
dataset_inputs:mul(1/std)


test_inputs = torch.Tensor(10000, 1, 28, 28)
--test_inputs = torch.Tensor(10000, 28, 28)
test_outputs = torch.Tensor(10000)

for i=1,testset.size do
    test_outputs[i] = testset[i].y + 1
    test_inputs[i] = testset[i].x
end

test_inputs:add(-mean)
test_inputs:mul(1/std)


if cuda then
  dataset_inputs = dataset_inputs:cuda()
  dataset_outputs = dataset_outputs:cuda()
  test_outputs = test_outputs:cuda()
  test_inputs = test_inputs:cuda()
end

print('dataset loaded')


-- define model to train
model = nn.Sequential()

if opt.model == 'linear' then
    model:add(nn.Reshape(784))
    model:add(nn.Linear(784, 10))
elseif opt.model == 'mlp' then
    model:add(nn.Reshape(784))
    model:add(nn.Linear(784, 300))
    model:add(nn.Tanh())
    model:add(nn.Linear(300, 10))
elseif opt.model == 'convnet' then
    model:add(nn.SpatialConvolution(1, 50, 5, 5, 1, 1, 0, 0))
    model:add(nn.SpatialBatchNormalization(50))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))
    model:add(nn.SpatialConvolution(50, 50, 5, 5, 1, 1, 0, 0))
    model:add(nn.SpatialBatchNormalization(50))
    model:add(nn.ReLU(true))
    --model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))
    model:add(nn.Reshape(8*8*50))
    model:add(nn.Linear(8*8*50, 160)) 
    model:add(nn.ReLU()) 
    model:add(nn.Linear(160, 10))  
    --model:add(nn.SpatialConvolutionMM(1, 20, 5, 5, 1, 1, 0, 0))
    --model:add(nn.SpatialBatchNormalization(20))
    --model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))
    --model:add(nn.SpatialConvolutionMM(20, 50, 5, 5, 1, 1, 0, 0))
    --model:add(nn.SpatialBatchNormalization(50))
    --model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))
    --model:add(nn.Reshape(4*4*50))
    --model:add(nn.Linear(4*4*50, 500)) 
    --model:add(nn.ReLU()) 
    --model:add(nn.Linear(500, 10))  
else
    print("unknown model")
    error()
end









model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
if cuda then
  model:cuda()
  criterion:cuda()
  -- Use cuDNN if available
  if hasCudnn then
    cudnn.convert(model, cudnn)
  end
end



x, dl_dx = model:getParameters()

feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end

    dl_dx:zero()

    local loss_x = criterion:forward(model:forward(inputs), targets)
    model:backward(inputs, criterion:backward(model.output, targets))

    return loss_x, dl_dx
end

sgd_params = {
    learningRate = opt.learningrate,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}

--epochs = 1e2
epochs = opt.epochs
model:training()
for i = 1,epochs do
    current_loss = 0

    for j = 1,(#dataset_inputs)[1],batchsize do
        inputs = dataset_inputs:narrow(1, j, batchsize)
        targets = dataset_outputs:narrow(1, j, batchsize)



        --inputs = torch.Tensor(batchsize, 1, 28, 28)
        --targets = torch.Tensor(batchsize)
        --local k = 1
        --for l=j,math.min(j+batchsize-1, dataset_inputs:size(1)) do
        --    inputs[k] = dataset_inputs[l]
        --    targets[k] = dataset_outputs[l]
        --    k = k + 1
        --end


        _, fs = optim.sgd(feval,x,sgd_params)
        current_loss = current_loss + fs[1] * batchsize
    end

    current_loss = current_loss / (#dataset_inputs)[1]
    print('epoch = ' .. i .. ' of ' .. epochs .. ' current loss = ' .. current_loss)
    --[[
    print('=====Testing the model======')

    function predict(x)
        --local input = torch.Tensor(1,1,28,28)
        --input[1] = x
        --local logProbs = model:forward(input)
        local logProbs = model:forward(x)
        local probs = torch.exp(logProbs[1])
        maxs, indices = torch.max(probs, 1)
        return indices
    end

    num = 0
    for j = 1,(#test_inputs)[1] do
    --for i = 1,10 do
        predict_label = predict(test_inputs:narrow(1,j,1))
        predict_label = predict_label:squeeze()
        --print(predict_label)
        --print(test_outputs[i])
        if predict_label ~= test_outputs[j] then
            num = num + 1
        end
    end
    
    print(num / (#test_inputs)[1])
    ]]


end

print('=====Testing the model======')

function predict(x)
    --local input = torch.Tensor(1,1,28,28)
    --input[1] = x
    --local logProbs = model:forward(input)
    local logProbs = model:forward(x)
    local probs = torch.exp(logProbs[1])
    maxs, indices = torch.max(probs, 1)
    return indices
end

model:evaluate()
num = 0
for i = 1,(#test_inputs)[1] do
--for i = 1,10 do
    predict_label = predict(test_inputs:narrow(1,i,1))
    predict_label = predict_label:squeeze()
    --print(predict_label)
    --print(test_outputs[i])
    if predict_label ~= test_outputs[i] then
        num = num + 1
    end
end

print(num / (#test_inputs)[1])









