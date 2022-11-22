# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models



def train(args, device, train_loader, traintest_loader, test_loader,recorder):
    torch.manual_seed(42)
    
    for trial in range(1,args.trials+1):
        
        # Network topology
        model = models.SRNN(n_in=args.n_inputs,
                            n_rec=args.n_rec,
                            n_out=args.n_classes,
                            n_t=args.n_steps,
                            thr=args.threshold,
                            tau_m=args.tau_mem,
                            tau_o=args.tau_out,
                            b_o=args.bias_out,
                            gamma=args.gamma,
                            dt=args.dt,
                            model=args.model,
                            classif=args.classif,
                            w_init_gain=args.w_init_gain,
                            lr_layer=args.lr_layer_norm,
                            t_crop=args.delay_targets,
                            visualize=args.visualize,
                            visualize_light=args.visualize_light,
                            device=device)

        # Use CUDA for GPU-based computation if enabled
        if args.cuda:
            model.cuda()
        
        # Initial monitoring
        if (args.trials > 1):
            print('\nIn trial {} of {}'.format(trial,args.trials))
        if (trial == 1):
            print("=== Model ===" )
            print(model)
        
        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        
        # Loss function (only for performance monitoring purposes, does not influence learning as e-prop learning is hardcoded)
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")
        
        # Training and performance monitoring
        print("\n=== Starting model training with %d epochs:\n" % (args.epochs,))        
        for epoch in range(1, args.epochs + 1):
            print("\t Epoch "+str(epoch)+"...")
            #Training:
            do_epoch(args, True, model, device, train_loader,test_loader, optimizer, loss, 'train',recorder)           # Will display the average accuracy on the training set during the epoch (changing weights)
            #Check performance on the training set and on the test set:
            if not args.skip_test:
                #do_epoch(args, False, model, device, traintest_loader, optimizer, loss, 'train') # Uncomment to display the final accuracy on the training set after the epoch (fixed weights)
                do_epoch(args, False, model, device, test_loader,test_loader, optimizer, loss, 'test',recorder)


def do_epoch(args, do_training, model, device, loader, test_loader,optimizer, loss_fct, benchType,recorder):
    model.eval()    # This implementation does not rely on autograd, learning update rules are hardcoded
    score = 0
    loss = 0
    batch = args.batch_size if (benchType == 'train') else args.test_batch_size
    length = args.full_train_len if (benchType == 'train') else len(loader)* [x for x in loader][0][0].shape[0] #args.full_test_len
    with torch.no_grad():   # Same here, we make sure autograd is disabled
        
        # For each batch
        for batch_idx, (data, label) in enumerate(loader):
            if do_training and batch_idx and batch_idx%20==0:
                do_duringtest(score,batch_idx,batch,loss,model,device,args,test_loader,optimizer,loss_fct,recorder)


            data, label = data.to(device), label.to(device)
            if args.classif:    # Do a one-hot encoding for classification
                targets = torch.zeros(label.shape, device=device).unsqueeze(-1).expand(-1,-1,args.n_classes).scatter(2, label.unsqueeze(-1), 1.0).permute(1,0,2)
            else:
                targets = label.permute(1,0,2)

            # Evaluate the model for all the time steps of the input data, then either do the weight updates on a per-timestep basis, or on a per-sample basis (sum of all per-timestep updates).
            optimizer.zero_grad()
            output = model(data.permute(1,0,2), targets, do_training)
            if do_training:
                optimizer.step()
                model.w_rec.data = model.w_rec.data * (1 - torch.eye(model.n_rec,device=device))

                
            # Compute the loss function, inference and score
            if args.delay_targets:
                loss += loss_fct[0](output[-args.delay_targets:], loss_fct[1](targets[-args.delay_targets:]), reduction='mean')
            else:
                loss += loss_fct[0](output, loss_fct[1](targets), reduction='mean')
            if args.classif:
                if args.delay_targets:
                    inference = torch.argmax(torch.sum(output[-args.delay_targets:],axis=0),axis=1)
                    score += torch.sum(torch.eq(inference,label[:,0]))
                else:
                    inference = torch.argmax(torch.sum(output,axis=0),axis=1)
                    score += torch.sum(torch.eq(inference,label[:,0]))




    if benchType == "train" and do_training:
        info = "on training set (while training): "
    elif benchType == "train":
        info = "on training set                 : "
    elif benchType == "test":
        info = "on test set                     : "

    if args.classif:
        print("\t\t Score "+info+str(score.item())+'/'+str(length)+' ('+str(score.item()/length*100)+'%), loss: '+str(loss.item()))
    else:
        print("\t\t Loss "+info+str(loss.item()))

    if benchType == "test" and do_training==False:
        return score.item() / length * 100
            
def do_duringtest(score,batch_idx,batch,loss,model,device,args,test_loader,optimizer,loss_fct,recorder):
    info = "on training set (while training): "

    if args.classif:
        print("\t\t Score " + info + str(score.item()) + '/' + str((batch_idx) * batch) + ' (' + str(
            score.item() / ((batch_idx) * batch) * 100) + '%), loss: ' + str(loss.item()))
    else:
        print("\t\t Loss " + info + str(loss.item()))

    print("on test set                     : ")
    test_accuracy=do_epoch(args, False, model, device, test_loader, test_loader, optimizer, loss_fct, 'test',recorder)



    print('Task on 1  :')
    task_accuracy=do_epoch(args, False, model, device, [i for i in test_loader][0:5], test_loader, optimizer, loss_fct, 'test',recorder)
    recorder.record_spike_v(batch_idx * batch,1,test_accuracy,task_accuracy,model.z[:,-1,:],model.v[:,-1,:])

    print('Task on 2  :')
    task_accuracy=do_epoch(args, False, model, device, [i for i in test_loader][5:10], test_loader, optimizer, loss_fct,
             'test',recorder)
    recorder.record_spike_v(batch_idx * batch, 2, test_accuracy, task_accuracy, model.z[:,-1,:],model.v[:,-1,:])

    print('Task on 3  :')
    task_accuracy=do_epoch(args, False, model, device, [i for i in test_loader][10:15], test_loader, optimizer, loss_fct,
             'test',recorder)
    recorder.record_spike_v(batch_idx * batch, 3, test_accuracy, task_accuracy, model.z[:,-1,:],model.v[:,-1,:])

    print('Task on 4  :')
    task_accuracy=do_epoch(args, False, model, device, [i for i in test_loader][15:20], test_loader, optimizer, loss_fct,
             'test',recorder)
    recorder.record_spike_v(batch_idx * batch, 4, test_accuracy, task_accuracy, model.z[:,-1,:],model.v[:,-1,:])

    #record model
    if batch_idx * batch==1200 or batch_idx * batch==1350 or batch_idx * batch==1500 or batch_idx * batch==1700 or batch_idx * batch==1850 or batch_idx * batch==2000:
        recorder.record_model(batch_idx * batch,model)