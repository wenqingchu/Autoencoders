local nn = require 'nn'

local Model = {}

function Model:createAutoencoder(X)
  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, 1, X:size(2), X:size(3)))
  self.encoder:add(nn.SpatialConvolution(1, 50, 5, 5, 1, 1, 0, 0))
  self.encoder:add(nn.SpatialBatchNormalization(50))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
  self.encoder:add(nn.SpatialConvolution(50, 50, 5, 5, 1, 1, 0, 0))
  self.encoder:add(nn.SpatialBatchNormalization(50))
  self.encoder:add(nn.ReLU(true))


  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.SpatialFullConvolution(50, 50, 3, 3, 2, 2, 1, 1))
  self.encoder:add(nn.SpatialBatchNormalization(50))
  self.decoder:add(nn.ReLU(true))

  self.decoder:add(nn.SpatialFullConvolution(50, 1, 4, 4, 2, 2, 2, 2))

  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model
