def train(num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, eval_loader, test_loader = get_dataset(dataDir='effnetb1_30_train_prob.csv')
    model = GraphOutModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    total_eval_loss  = 0
    correct_domain   = 0
    correct_category = 0
    total            = 0
    for epoch in range(num_epochs):
############################################train phase############################################
        model.train()
        total_loss = 0
        correct_domain   = 0
        correct_category = 0
        total            = 0
        count            = 0
        for images, domains, categories, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Training'):
            images, domains, categories, labels = images.to(device), domains.to(device), categories.to(device), labels.to(device)

            logits = model(images)
            probs  = torch.softmax(logits, dim=2)
            _, domainOut, categoryOut = get_correct(probs)
            domainLossInAtProbs,     domainLossInAtLabels = torch.sum(probs, dim=1), torch.sum(labels, dim=1)
            categoryLossInAtProbs, categoryLossInAtLabels = torch.sum(probs, dim=2), torch.sum(labels, dim=2)
            loss = criterion(domainLossInAtProbs, domainLossInAtLabels) + criterion(categoryLossInAtProbs, categoryLossInAtLabels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss       += loss.item()
            correct_domain   += (domainOut.to(device)   == domains).float().sum().item()
            correct_category += (categoryOut.to(device) == categories).float().sum().item()
            #print('\nd',domainOut)
            #print('D',domains)
            total += images.size(0)
            if count % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}, Domain Accuracy: {100 * (correct_domain / total):.2f}%, Category Accuracy: {100 * (correct_category / total):.2f}%")
            if count % 300 == 0:
                print('\n: correct label pos : ', domainOut[0], categoryOut[0])
                for i in range(probs.size(dim=1)):
                    for j in range(probs.size(dim=2)):
                        print(f"{probs[0][i][j]:.2f}",end=' ')
                    print()
                print()
            count += 1
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}")

############################################validation phase############################################
        model.eval()
        total_eval_loss  = 0
        correct_domain   = 0
        correct_category = 0
        total            = 0
        count = 0
        with torch.no_grad():
            for images, domains, categories, labels in tqdm(eval_loader, desc=f'Epoch {epoch + 1} Evaluation'):
                images, domains, categories, labels = images.to(device), domains.to(device), categories.to(device), labels.to(device)

                logits = model(images)
                inferenceOut = torch.softmax(logits, dim=2)
                _, domainOut, categoryOut = get_correct(inferenceOut)

                eval_loss = criterion(logits, labels)
                total_eval_loss += eval_loss.item()

                # 정확도 계산
                correct_domain   += (domainOut.to(device)   == domains).float().sum().item()
                correct_category += (categoryOut.to(device) == categories).float().sum().item()
                total += images.size(0)
                if count % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}, Domain Accuracy: {100 * (correct_domain / total):.2f}%, Category Accuracy: {100 * (correct_category / total):.2f}%")
                count += 1
            print(f"\nEpoch {epoch + 1}/{num_epochs}, Eval Loss: {total_eval_loss / len(eval_loader)}, Domain Accuracy: {100 * (correct_domain / total):.2f}%, Category Accuracy: {100 * (correct_category / total):.2f}%")