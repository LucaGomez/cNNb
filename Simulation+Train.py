import camb
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
from utils import *
from utils import train_set_generator, valid_set_generator, test_set_generator
from utils import cmb_temperature_map, dust_temperature_map, frequency_cont_maps


nside = 512  # Resolución del mapa
npix = hp.nside2npix(nside)  # Número total de píxeles

# Configurar las coordenadas de la región
l_center, b_center = 316, -56
delta_l, delta_b = 40, 40

l_min, l_max = l_center - delta_l / 2, l_center + delta_l / 2
b_min, b_max = b_center - delta_b / 2, b_center + delta_b / 2

# Coordenadas de los píxeles en orden NESTED
# Primero generamos índices NESTED y convertimos a coordenadas angulares
nested_indices = np.arange(npix)
theta, phi = hp.pix2ang(nside, nested_indices, nest=True)  # Coordenadas esféricas (en radianes)
l = np.degrees(phi)  # Longitud galáctica
b = 90 - np.degrees(theta)  # Latitud galáctica

# Aplicar la máscara de la región
region_mask = (l >= l_min) & (l <= l_max) & (b >= b_min) & (b <= b_max)

# Obtener los índices NESTED de los píxeles en la región
pixels_in_region = nested_indices[region_mask]
mapa = np.full(npix, 0.5)  # Fondo uniforme con valor 0.5
'''THIS IS ONLY VALID FOR nside=512
'''
indices_interes = pixels_in_region[657:66193]
T0 = 2.726 # K

pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)

n_maps_train = 20
n_maps_valid = 10
n_maps_test = 5

freqs = np.array([100,150,220,300])

train_set_generator(n_maps_train,nside,pars,freqs,indices_interes)
valid_set_generator(n_maps_valid,nside,pars,freqs,indices_interes)
test_set_generator(n_maps_test,nside,pars,freqs,indices_interes)


# Inicializa la U-Net con profundidad ajustada para entradas 256x256
net = UNet(
    num_classes=1,       # Número de clases en tu tarea
    in_channels=4,       # Número de canales en la entrada
    depth=4,             # Profundidad ajustada para inputs 256x256
    start_filts=64,      # Filtros iniciales
    up_mode='transpose', # Tipo de upsampling
    merge_mode='concat'  # Tipo de fusión
)


batch_size = 4
iteration_n = 50
lr = 1e-3
loss_func = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
repeat_n = 4
np.random.seed(1000)
loss_all = []
loss_val = []


for iteration in range(1, iteration_n+1):
    print(iteration)
    os.chdir('Train_set')
    map_nums = np.random.choice(n_maps_train, batch_size, replace=False)
    xx = arrange_input(map_nums)
    
    yy = arrange_target(map_nums)
        
    if iteration%(500+1)==0:
        print ('X_mean:%.4f'%(xx.mean()),'X_min:%.4f'%(xx.min()),'X_max:%.4f'%(xx.max()))
        print ('y_mean:%.4f'%(yy.mean()),'y_min:%.4f'%(yy.min()),'y_max:%.4f'%(yy.max()))
    xx = Variable(xx); yy = Variable(yy, requires_grad=False)
    
    repeat_n = repeat_n
    for t in range(repeat_n):
        predicted = net(xx)
        if t+1==repeat_n and iteration%(500+1)==0:
            print ('p_mean:%.4f'%(predicted.data.mean()),'p_min:%.4f'%(predicted.data.min()),'p_max:%.4f \n'%(predicted.data.max()))
        loss = loss_func(predicted, yy)
        loss_all.append(loss.item())
        print(loss.item())
        if t+1==repeat_n and iteration%50==0:
            print ('(iteration:%s/%s; loss:%.5f; lr:%.8f)'%(iteration, iteration_n, loss.item(), optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    os.chdir('../Valid_set')
    map_nums = np.random.choice(n_maps_valid, batch_size, replace=False)
    xx = arrange_input(map_nums)
    yy = arrange_target(map_nums)
    xx = Variable(xx); yy = Variable(yy, requires_grad=False)
    predicted = net(xx)
    loss_valid = loss_func(predicted, yy)
    print('loss valid= '+str(loss_valid.item()))
    for i in range(repeat_n):
        loss_val.append(loss_valid.item())
    os.chdir('..')
    
plt.plot(loss_all, label = 'Train loss')
plt.plot(loss_val, label = 'Valid loss')
plt.yscale('log')
plt.xlabel('Iterations')
plt.title('Lr = '+str(lr))
plt.savefig('loss_cNNb_lr3.png')
plt.legend()
plt.show()

model_path = "unet_model_lr3.pth"

# Guardar los pesos del modelo
torch.save(net.state_dict(), model_path)


batch_size = 4
iteration_n = 10
lr = 1e-4
loss_func = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
repeat_n = 4
np.random.seed(42)
loss_all = []
loss_val = []


for iteration in range(1, iteration_n+1):
    print(iteration)
    os.chdir('Train_set')
    map_nums = np.random.choice(n_maps_train, batch_size, replace=False)
    xx = arrange_input(map_nums)
    
    yy = arrange_target(map_nums)
        
    if iteration%(500+1)==0:
        print ('X_mean:%.4f'%(xx.mean()),'X_min:%.4f'%(xx.min()),'X_max:%.4f'%(xx.max()))
        print ('y_mean:%.4f'%(yy.mean()),'y_min:%.4f'%(yy.min()),'y_max:%.4f'%(yy.max()))
    xx = Variable(xx); yy = Variable(yy, requires_grad=False)
    
    repeat_n = repeat_n
    for t in range(repeat_n):
        predicted = net(xx)
        if t+1==repeat_n and iteration%(500+1)==0:
            print ('p_mean:%.4f'%(predicted.data.mean()),'p_min:%.4f'%(predicted.data.min()),'p_max:%.4f \n'%(predicted.data.max()))
        loss = loss_func(predicted, yy)
        loss_all.append(loss.item())
        print(loss.item())
        if t+1==repeat_n and iteration%50==0:
            print ('(iteration:%s/%s; loss:%.5f; lr:%.8f)'%(iteration, iteration_n, loss.item(), optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    os.chdir('../Valid_set')
    map_nums = np.random.choice(n_maps_valid, batch_size, replace=False)
    xx = arrange_input(map_nums)
    yy = arrange_target(map_nums)
    xx = Variable(xx); yy = Variable(yy, requires_grad=False)
    predicted = net(xx)
    loss_valid = loss_func(predicted, yy)
    print('loss valid= '+str(loss_valid.item()))
    for i in range(repeat_n):
        loss_val.append(loss_valid.item())
    os.chdir('..')
    
plt.plot(loss_all, label = 'Train loss')
plt.plot(loss_val, label = 'Valid loss')
plt.yscale('log')
plt.xlabel('Iterations')
plt.title('Lr = '+str(lr))
plt.savefig('loss_cNNb_lr3.png')
plt.legend()
plt.show()

model_path = "unet_model_lr3.pth"

# Guardar los pesos del modelo
torch.save(net.state_dict(), model_path)



os.chdir('Test_set')
for i in range(n_maps_test):
    xx = arrange_input(np.array([i]))
    target = arrange_target(np.array([i]))
    xx = Variable(xx)
    predicted = net(xx)
    residual = predicted - target
    
    orig_map = xx[0,0,:,:].detach().numpy()
    rec_map = predicted[0,0,:,:].detach().numpy()
    targ_map = target[0,0,:,:].detach().numpy()
    res_map = residual[0,0,:,:].detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 fila, 3 columnas
    # Primer gráfico con imshow
    im1 = axes[0].imshow(rec_map, cmap='viridis')
    axes[0].set_title('Predicted')
    fig.colorbar(im1, ax=axes[0], orientation='vertical')  # Barra de color opcional

    # Segundo gráfico con imshow
    im2 = axes[1].imshow(targ_map, cmap='viridis')
    axes[1].set_title('Target')
    fig.colorbar(im2, ax=axes[1], orientation='vertical')
    
    im3 = axes[2].imshow(res_map, cmap='viridis')
    axes[2].set_title('Residual')
    fig.colorbar(im3, ax=axes[2], orientation='vertical')


    # Ajustar el espacio entre los subplots
    plt.tight_layout()

    # Mostrar la figura
    #plt.savefig('Test_custom'+str(i)+'.pdf')
    plt.show()
    
    cmb_array_recov = nestedMap2nestedArray(rec_map,nside)           #Define un nuevo array a partir del box.
    map_cmb_new=np.zeros(npix)                             #Define un mapa vacio para llenarlo
    map_cmb_new[indices_interes]=cmb_array_recov           #Llena el mapa vacio con el array recuperado
    map_cmb_new_ring = hp.reorder(map_cmb_new, n2r=True)   #Aplica el reordenamiento nest--->ring
    plt.figure()
    hp.mollview(map_cmb_new_ring, title='Predicted map')                          #Plotea el parche recuperado en ordenamiento ring
    #plt.savefig('Mollview_predicted'+str(i)+'.pdf')
    plt.show()
    
    cmb_array_targ = nestedMap2nestedArray(targ_map,nside)           #Define un nuevo array a partir del box.
    map_cmb_new_targ=np.zeros(npix)                             #Define un mapa vacio para llenarlo
    map_cmb_new_targ[indices_interes]=cmb_array_targ           #Llena el mapa vacio con el array recuperado
    map_cmb_new_targ_ring = hp.reorder(map_cmb_new_targ, n2r=True)   #Aplica el reordenamiento nest--->ring
    plt.figure()
    hp.mollview(map_cmb_new_targ_ring, title='Target map')                          #Plotea el parche recuperado en ordenamiento ring
    #plt.savefig('Mollview_target'+str(i)+'.pdf')
    plt.show()
    
os.chdir('..')
