
from src.models.netunits import *
from logzero import logger

class Net(nn.Module):
    def __init__(self, out_ch, d, embedding_dim, with_embedding, embedding_first_layer, image_type):
        super(Net, self).__init__()
        if image_type == 'huipiao':
            param1 = 1024
            param2 = 2048
        else:
            param1 = 512
            param2 = 1024
        self.with_embedding = with_embedding
        self.embedding_first_layer = embedding_first_layer
        self.inc = InConv(3, 128, d)
        self.inc2 = InConv(3 + embedding_dim, 128, d)
        self.down1 = Down(128, 256, d)
        self.down2 = Down(256, 512, d)
        self.down3 = Down(512, param1, d)
        self.down4 = Down(param1, param1, d)
        self.upsamp = Upsamp()
        self.resize = Resize()
        self.up1 = Up(param2, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.outc1 = OutConv(64, out_ch)
        self.outc2 = OutConv(embedding_dim + 64, out_ch)

    def forward(self, x, embedding_layer):
        # logger.info(type(x))
        temp_x = x
        if self.with_embedding and self.embedding_first_layer:
            x1 = self.resize(x, embedding_layer)
            x1 = torch.cat((x1, embedding_layer), dim=1)  # 3 + 128
            x1 = self.inc2(x1)
        else:
            x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.upsamp(x5)
        x = self.resize(x, x4)
        x = self.up1(x, x4)         # 256

        x = self.upsamp(x)
        x = self.resize(x, x3)
        Rec_x3 = x
        x = self.up2(x, x3)     # 128

        x = self.upsamp(x)
        x = self.resize(x, x2)
        Rec_x2 = x
        x = self.up3(x, x2)     # 64

        x = self.upsamp(x)
        x = self.resize(x, x1)
        Rec_x1 = x
        x = self.up4(x, x1)     # 64

        Rec_x = self.outc1(x)         # 3      for reconstruction
        Rec_x = self.resize(Rec_x, temp_x)

        x = self.upsamp(x5)
        x = self.resize(x, x4)
        x = self.up1(x, x4)  # 256

        x = self.upsamp(x)
        x = self.resize(x, x3)
        x = self.up2(x, x3)  # 128

        x = self.upsamp(x)
        x = self.resize(x, x2)
        x = self.up3(x, x2)  # 64

        x = self.upsamp(x)
        x = self.resize(x, x1)
        x = self.up4(x, x1)  # 64

        if self.with_embedding and not self.embedding_first_layer:
            x = self.resize(x, embedding_layer)
            x = torch.cat((x, embedding_layer), dim=1)   # 64 + 128
            Seg_x = self.outc2(x)  # 3            # for segmentation
        else:
            Seg_x = self.outc1(x)

        # Seg_x = torch.sigmoid(Seg_x)
        return Seg_x, Rec_x, Rec_x1, x1, Rec_x2, x2, Rec_x3, x3


def create_model(cfg, args):
    return Net(cfg.MODEL.NUM_CLASSES, args.dilation, args.embedding_dim, args.with_embedding, args.embedding_first_layer, cfg.DATASET.TYPE)


if __name__ == '__main__':
    while 1:
        scale = 1
        image = torch.randn((1, 3, int(663 * scale), int(512 * scale))).cuda()
        embedding = torch.randn((1, 128, int(663 * scale), int(512 * scale))).cuda()
        model = Net(3, 3).cuda()

        out, Rec_x, Rec_x1, x1, Rec_x2, x2, Rec_x3, x3 = model(image, embedding)
        print(out.size(), x1.size(), 'x1', Rec_x1.size(), 'x2',  x2.size(), Rec_x2.size(), 'x3',  x3.size(), Rec_x3.size(), Rec_x.size())

