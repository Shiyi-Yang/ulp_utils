def conv(spec,delta,area):
    import imageio
    f0 = linspace(3,8,spec.shape[1])
    image_arr = []
    for row in range(spec.shape[0]):
        #spec[delta[i]]
        spec[max(row-psf_my,0):min(spec.shape[0],row+psf.shape[0]-psf_my)] += area[row] * psf[max(psf_my-row,0):min(psf.shape[0],psf_my+spec.shape[0]-row),
                                                                                  psf_mx - delta[row]:psf_mx + spec.shape[1]-delta[row]] 
        if (row % 20 == 0):
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(f0,spec[550])
            ax.set_title('spectrum at height %d km from %dth height contribution'%(height[550],height[row]))
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image_arr.append(image)
            fig.clf()
    imageio.mimsave('/home/yang158/test.gif', image_arr,fps=5)
    return spec

#figname = 'fig_{}.png'.format(i)
#        dest = os.path.join(path, figname)
#        plt.savefig(dest)  # write image to file
