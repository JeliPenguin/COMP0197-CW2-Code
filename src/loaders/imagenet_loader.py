import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader



mean = torch.Tensor([0.485, 0.456, 0.406])
    
std = torch.Tensor([0.229, 0.224, 0.225])

def transform_image(image, args, mean, std):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(lambda x: x.convert("RGB")),  # Ensure image is RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(image)


def get_hugging_face_loaders(args):
    def collate_fn(batch,args,mean,std):
        # Set up the transformation: convert all images to 3 channels, resize, and convert to tensor

        images, labels = [], []
        for item in batch:
            image = transform_image(item['image'],args, mean, std)
            label = torch.tensor(item['label'], dtype=torch.long)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.stack(labels)
    
    # Ensure the dataset is properly loaded with streaming set to True
    if args.imagenet:
        print("Using ImageNet1k")
        train_dataset = load_dataset("imagenet-1k", split="train", streaming=True,trust_remote_code=True)
        test_dataset = load_dataset("imagenet-1k", split="test", streaming=True,trust_remote_code=True)
        
    elif args.partial_imagenet or args.train_mode == "pruned_pretrain":
        print("Using partial ImageNet1k")
        return get_hugging_face_partial_imagenet_loaders(args)
    else:
        print("Using tiny imagenet")
        train_dataset = load_dataset('Maysee/tiny-imagenet', split="train", streaming=True,trust_remote_code=True)
        test_dataset = load_dataset("Maysee/tiny-imagenet", split="valid", streaming=True,trust_remote_code=True)

    # Setup DataLoader with the custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, args,mean,std)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, args,mean,std)
    )

    return train_loader, test_loader, mean, std


def get_hugging_face_partial_imagenet_loaders(args):
    def collate_fn_batch(batch):
        images = []
        labels = []

        for item in batch:
            image = item['image']
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float)

            images.append(image)

            # Handling labels
            label = item['label']
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)

            labels.append(label)

        # Stack all images and labels into tensors
        images = torch.stack(images)
        labels = torch.stack(labels)

        return images, labels

    train_dataset, test_dataset = load_and_process_datasets(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_batch)

    return train_loader, test_loader, mean, std


def load_and_process_datasets(args,label_filter=[
        1,      # Goldfish, Carassius auratus
        12,     # House finch, linnet, Carpodacus mexicanus
        87,     # African grey, African gray, Psittacus erithacus
        88,     # Macaw
        89,     # Sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
        90,     # Lorikeet
        91,     # Coucal
        151,    # Chihuahua
        152,    # Japanese spaniel
        153,    # Maltese dog, Maltese terrier, Maltese
        154,    # Pekinese, Pekingese, Peke
        155,    # Shih-Tzu
        156,    # Blenheim spaniel
        157,    # Papillon
        158,    # Toy terrier
        159,    # Rhodesian ridgeback
        160,    # Afghan hound, Afghan
        161,    # Basset, basset hound
        162,    # Beagle
        163,    # Bloodhound, sleuthhound
        164,    # Bluetick
        165,    # Black-and-tan coonhound
        166,    # Walker hound, Walker foxhound
        167,    # English foxhound
        168,    # Redbone
        169,    # Borzoi, Russian wolfhound
        170,    # Irish wolfhound
        171,    # Italian greyhound
        172,    # Whippet
        173,    # Ibizan hound, Ibizan Podenco
        174,    # Norwegian elkhound, elkhound
        175,    # Otterhound, otter hound
        176,    # Saluki, gazelle hound
        177,    # Scottish deerhound, deerhound
        178,    # Weimaraner
        179,    # Staffordshire bull terrier, Staffordshire bullterrier
        180,    # American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
        181,    # Bedlington terrier
        182,    # Border terrier
        183,    # Kerry blue terrier
        184,    # Irish terrier
        185,    # Norfolk terrier
        186,    # Norwich terrier
        187,    # Yorkshire terrier
        188,    # Wire-haired fox terrier
        189,    # Lakeland terrier
        190,    # Sealyham terrier, Sealyham
        191,    # Airedale, Airedale terrier
        192,    # Cairn, cairn terrier
        193,    # Australian terrier
        194,    # Dandie Dinmont, Dandie Dinmont terrier
        195,    # Boston bull, Boston terrier
        196,    # Miniature schnauzer
        197,    # Giant schnauzer
        198,    # Standard schnauzer
        199,    # Scotch terrier, Scottish terrier, Scottie
        200,    # Tibetan terrier, chrysanthemum dog
        201,    # Silky terrier, Sydney silky
        202,    # Soft-coated wheaten terrier
        203,    # West Highland white terrier
        204,    # Lhasa, Lhasa apso
        205,    # Flat-coated retriever
        206,    # Curly-coated retriever
        207,    # Golden retriever
        208,    # Labrador retriever
        209,    # Chesapeake Bay retriever
        245,    # French bulldog
        250,    # Siberian husky
        251,    # Dalmatian, coach dog, carriage dog
        259,    # Pomeranian
        260,    # Chow, chow chow
        261,    # Keeshond
        262,    # Brabancon griffon
        263,    # Pembroke, Pembroke Welsh corgi
        264,    # Cardigan, Cardigan Welsh corgi
        265,    # Toy poodle
        266,    # Miniature poodle
        267,    # Standard poodle
        268,    # Mexican hairless
        281,    # Tabby, tabby cat
        282,    # Tiger cat
        283,    # Persian cat
        284,    # Siamese cat, Siamese
        285,    # Egyptian cat
        # 330,    # Rabbit, wood rabbit, cottontail, cottontail rabbit
        # 333,    # Hamster
        # 338     # Guinea pig, Cavia cobaya
    ]):

    train_dataset = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)
    test_dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)

    print('Applying transformations and filtering datasets.')
    
    # transform_lambda = lambda x: {'image': transform_image(x['image'], args, mean, std), 'label': x['label']}
    def batch_transform(batch):
        # Apply the transform_image to each image in the batch
        batch['image'] = [transform_image(image, args, mean, std) for image in batch['image']]

        # The labels don't need transformation, just pass them through
        batch['label'] = batch['label']
        return batch

    # Apply the transformations and filter the datasets
    train_dataset = train_dataset.filter(lambda x: x['label'] in label_filter).map(batch_transform, batched=True, batch_size=args.batch_size)
    test_dataset = test_dataset.filter(lambda x: x['label'] in label_filter).map(batch_transform, batched=True, batch_size=args.batch_size)

    return train_dataset, test_dataset


def calculate_mean_std(dataset, first=3000):
    
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    # use only m examples to compute pixel means and stds
    m = first if len(dataset) > first else len(dataset)
    subset_indices = torch.randperm(len(dataset))[:m]  
    dataset_for_stats = Subset(dataset, subset_indices)
    # Manually iterate through the dataset
    for data, _ in dataset_for_stats:
        # Flatten the channel dimension
        data = data.view(3, -1)
        channel_sum += data.sum(dim=1)
        channel_squared_sum += (data ** 2).sum(dim=1)
        num_pixels += data.shape[1]

    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean ** 2) ** 0.5 # var = E[X**2] - (E[X])**2

    return mean, std
