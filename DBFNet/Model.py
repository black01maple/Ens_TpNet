import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn

# 定义台风路径、风力信息的编码器参数
TC_Encoder_config = {
    'lstm': {
        'input_size': 6,
        'hidden_size': 32,
        'num_layers': 2,
        'bias': True,
        'batch_first': True,
        'dropout': 0
    },
    'fc': {
        'in_features': 32,
        'out_features': 2,
        'bias': True
    }
}

# 定义台风空间数据的编码器参数，此参数同时用于计算对应的解码器参数
Map_Encoder_config = {
    'conv': {
        'channels': [9, 16, 32, 64],
        'kernel': 3,
        'stride': 1,
        'padding': False,
        'bias': True,
    },
    'maxpool': {
        'kernel': 2,
        'stride': 2
    },
    'fc': {
        'features': [4608, 512, 128],
        'bias': True
    }
}


class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1, bn=True):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv_channel = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn = bn

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out2 = self.conv_channel(x)
            out = F.relu(out + out2)
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out2 = self.conv_channel(x)
            out = F.relu(out + out2)
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2, out_padding=0, bn=True):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True, output_padding=out_padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn = bn

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        return out


class Unet(nn.Module):
    def __init__(self, bn=True):
        super(Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(9, 8, bn=bn)
        self.layer2_conv = double_conv2d_bn(8, 16, bn=bn)
        self.layer3_conv = double_conv2d_bn(16, 32, bn=bn)
        self.layer4_conv = double_conv2d_bn(32, 64, bn=bn)
        self.layer5_conv = double_conv2d_bn(64, 128, bn=bn)
        self.layer6_conv = double_conv2d_bn(128, 64, bn=bn)
        self.layer7_conv = double_conv2d_bn(64, 32, bn=bn)
        self.layer8_conv = double_conv2d_bn(32, 16, bn=bn)
        self.layer9_conv = double_conv2d_bn(16, 8, bn=bn)
        self.layer10_conv = nn.Conv2d(8, 9, kernel_size=3, stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64, bn=bn)
        self.deconv2 = deconv2d_bn(64, 32, bn=bn)
        self.deconv3 = deconv2d_bn(32, 16, bn=bn)
        self.deconv4 = deconv2d_bn(16, 8, out_padding=1, bn=bn)

        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(nn.Linear(in_features=4608, out_features=512, bias=True),
                                nn.Linear(in_features=512, out_features=128, bias=True))

    def forward(self, x):
        x_shape = x.shape
        first = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4])
        conv1 = self.layer1_conv(first)
        pool1 = F.max_pool2d(conv1, 2)
        # print(pool1.shape)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        # print(pool2.shape)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        # print(pool3.shape)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        # print(pool4.shape)

        conv5 = self.layer5_conv(pool4)

        middle_shape = conv5.shape
        conv5_reshaped = conv5.reshape(x_shape[0], -1, middle_shape[1], middle_shape[2], middle_shape[3])
        EGPH = self.fc(self.flatten(conv5_reshaped)).unsqueeze(1)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)
        # print(conv6.shape)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)
        # print(conv7.shape)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)
        # print(conv8.shape)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = F.relu(self.layer10_conv(conv9))
        out = outp.reshape(x_shape)
        return EGPH, out


class TC_Encoder(nn.Module):
    def __init__(self, config):
        super(TC_Encoder, self).__init__()
        self.lstm = nn.LSTM(**config['lstm'])
        self.fc = nn.Linear(**config['fc'])
        self.lstm_input_size = config['lstm']['input_size']
        self.lstm_hidden_size = config['lstm']['hidden_size']
        self.lstm_num_layers = config['lstm']['num_layers']

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros([self.lstm_num_layers, batch_size, self.lstm_hidden_size])
        c0 = torch.zeros([self.lstm_num_layers, batch_size, self.lstm_hidden_size])
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
        y, (hn, cn) = self.lstm(x, (h0, c0))
        ETC = torch.mean(y, dim=1).unsqueeze(1)
        y = y.reshape(-1, self.lstm_hidden_size)
        y = self.fc(y)
        y = y.reshape(batch_size, -1, 2)
        return ETC, y, (hn, cn)


class Map_Encoder(nn.Module):
    def __init__(self, config):
        super(Map_Encoder, self).__init__()
        conv_config, maxpool_config, fc_config = config['conv'], config['maxpool'], config['fc']
        if conv_config['padding'] == True:
            padding = conv_config['kernel'] // 2
        else:
            padding = 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_config['channels'][0], conv_config['channels'][1], kernel_size=conv_config['kernel'],
                      stride=conv_config['stride'], padding=padding, bias=conv_config['bias']),
            nn.BatchNorm2d(conv_config['channels'][1]),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_config['channels'][1], conv_config['channels'][2], kernel_size=conv_config['kernel'],
                      stride=conv_config['stride'], padding=padding, bias=conv_config['bias']),
            nn.BatchNorm2d(conv_config['channels'][2]),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_config['channels'][2], conv_config['channels'][3], kernel_size=conv_config['kernel'],
                      stride=conv_config['stride'], padding=padding, bias=conv_config['bias']),
            nn.BatchNorm2d(conv_config['channels'][3]),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_config['kernel'], stride=maxpool_config['stride'],
                                    return_indices=True)
        layers = [nn.Flatten(start_dim=1)]
        in_features = fc_config['features'][0]
        for out_features in fc_config['features'][1:]:
            layers.append(nn.Linear(in_features, out_features, bias=fc_config['bias']))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Sigmoid())
            in_features = out_features
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        shape1 = y.shape
        y, ind1 = self.maxpool(y)
        y = self.conv2(y)
        shape2 = y.shape
        y, ind2 = self.maxpool(y)
        y = self.conv3(y)
        shape3 = y.shape
        y, ind3 = self.maxpool(y)
        # EGPH = self.fc(y)
        return y, [ind1, ind2, ind3], [shape1, shape2, shape3]


class Map_Decoder(nn.Module):
    def __init__(self, encoder_config):
        super(Map_Decoder, self).__init__()
        # 先计算一下decoder_config
        decoder_config = encoder_config
        decoder_config['conv']['channels'] = list(reversed(encoder_config['conv']['channels']))
        conv_config, maxpool_config = decoder_config['conv'], decoder_config['maxpool']
        if conv_config['padding'] == True:
            padding = conv_config['kernel'] // 2
        else:
            padding = 0
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(conv_config['channels'][0], conv_config['channels'][1],
                               kernel_size=conv_config['kernel'],
                               stride=conv_config['stride'], padding=padding, bias=conv_config['bias']),
            nn.BatchNorm2d(conv_config['channels'][1]),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(conv_config['channels'][1], conv_config['channels'][2],
                               kernel_size=conv_config['kernel'],
                               stride=conv_config['stride'], padding=padding, bias=conv_config['bias']),
            nn.BatchNorm2d(conv_config['channels'][2]),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(conv_config['channels'][2], conv_config['channels'][3],
                               kernel_size=conv_config['kernel'],
                               stride=conv_config['stride'], padding=padding, bias=conv_config['bias']),
            nn.BatchNorm2d(conv_config['channels'][3]),
            nn.ReLU())
        self.maxunpool = nn.MaxUnpool2d(kernel_size=maxpool_config['kernel'], stride=maxpool_config['stride'])

    def forward(self, x, ind, shape):
        y = self.maxunpool(x, ind[2], output_size=shape[2])
        y = self.conv1(y)
        y = self.maxunpool(y, ind[1], output_size=shape[1])
        y = self.conv2(y)
        y = self.maxunpool(y, ind[0], output_size=shape[0])
        y = self.conv3(y)
        return y


class Map_Branch_Model(nn.Module):
    def __init__(self, encoder_config):
        super(Map_Branch_Model, self).__init__()
        self.encoder = Map_Encoder(encoder_config)
        self.decoder = Map_Decoder(encoder_config)
        self.fc = self.encoder.fc

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        y, ind, shape = self.encoder(x)
        EGPH = self.fc(y.reshape(x_shape[0], -1))
        y = self.decoder(y, ind, shape)
        y = y.reshape(x_shape)
        EGPH = EGPH.unsqueeze(1)
        return EGPH, y


class DBF_Decoder(nn.Module):
    def __init__(self, config):
        super(DBF_Decoder, self).__init__()
        self.lstm = nn.LSTM(**config['lstm'])
        self.fc = nn.Linear(**config['fc'])

    def forward(self, ETC, EGPH, h0, c0):
        y0 = torch.zeros([ETC.shape[0], 1, 2])
        y = torch.zeros([ETC.shape[0], 1, 2])
        if torch.cuda.is_available():
            y0, y = y0.cuda(), y.cuda()
        hn, cn = h0, c0
        # print(ETC.shape, EGPH.shape)
        for i in range(1, 5):
            # print(y0.shape, ETC.shape, EGPH.shape)
            x = torch.concat([y0, ETC, EGPH], dim=2)
            x, (hn, cn) = self.lstm(x, (hn, cn))
            x = x.squeeze(1)
            x = self.fc(x)
            y0 = x.unsqueeze(1)
            y = torch.concat([y, y0], dim=1)
        y = y[:, 1:, :]
        return y


class DBFNet(nn.Module):
    def __init__(self, TC_Encoder_config, Map_Encoder_config):
        super(DBFNet, self).__init__()
        self.tc_encoder = TC_Encoder(TC_Encoder_config)
        # self.map_branch = Map_Branch_Model(Map_Encoder_config)
        self.map_branch = Unet(bn=True)
        # 通过两个Encoder的参数计算模型Decoder的参数
        DBF_Decoder_config = {
            'lstm': {
                'input_size': 2 + TC_Encoder_config['lstm']['hidden_size'] + Map_Encoder_config['fc']['features'][-1],
                'hidden_size': TC_Encoder_config['lstm']['hidden_size'],
                'num_layers': TC_Encoder_config['lstm']['num_layers'],
                'bias': True,
                'batch_first': True,
                'dropout': 0.3
            },
            'fc': {
                'in_features': TC_Encoder_config['lstm']['hidden_size'],
                'out_features': 2,
                'bias': True
            }
        }
        self.dbf_decoder = DBF_Decoder(DBF_Decoder_config)

    def pretrain_tc_forward(self, x):
        _, y, _ = self.tc_encoder(x)
        return y

    def pretrain_map_forward(self, x):
        _, m = self.map_branch(x)
        return m

    def forward(self, x_tc, x_map):
        etc, _, (hn, cn) = self.tc_encoder(x_tc)
        EGPH, m = self.map_branch(x_map)
        y = self.dbf_decoder(etc, EGPH, hn, cn)
        return y, m


'''
model = DBFNet(TC_Encoder_config, Map_Encoder_config)
x_tc = torch.rand(20, 5, 6)
x_map = torch.rand(20, 9, 97, 97)
y, m = model(x_tc, x_map)
print(y.shape, m.shape)
y = model.pretrain_tc_forward(x_tc)
print(y.shape)
y = model.pretrain_map_forward(x_map)
print(y.shape)
'''
