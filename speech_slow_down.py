import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import sounddevice as sd

#Ahigad Genisg & Omer Sela
# 316228022 & 316539535

# %% Functions
#calculate fourier coefficent
def FourierCoeffGen(signal):
    # TODO: Implement the FourierCoeffGen function.
    # This function compute signal's Fourier Coefficients.

    ############# Your code here ############
    FourierCoeff = []
    N = int(len(signal))
    w0 = 2 * np.pi / N
    temp = 0
    for k in range(N):
        temp = 0
        for n in range(N):
            expo = np.exp(-1j*k*w0*n)
            temp += signal[n]*expo

        temp = temp/N
        FourierCoeff.append(temp)
    #########################################

    return FourierCoeff

#calculate fourier series
def DiscreteFourierSeries(FourierCoeff):
    # TODO: Implement the FourierSeries function.
    # This function compute the Discrete Fourier Series from Fourier Coefficients.
    signal = []
    lenAk = int(len(FourierCoeff))
    N = lenAk
    w0 = 2 * np.pi / N

    for n in range(N):
        temp = 0
        for k in range(N):
            expo = np.exp(1j * w0 * k * n)
            temp += FourierCoeff[k]*expo
        signal.append(temp)

    ############# Your code here ############

    #########################################
    return signal
#speech slow down function
def func():
    # %% import wav file
    wav_path = "C:/Users/ahiga/Desktop/python Signals&Systems/fire.wav"  # Insert your path here, you can pick another wav file!
    signal, fs = sf.read(wav_path)
    signal = signal[:10 * fs]  # 10 secounds

    plt.figure(1)
    plt.title("Input signal Wave")
    plt.plot(signal)

    print(len(signal))
    # %% Parameters
    N = int(512)
    step = int(N / 4)
    kk = 0
    M = 3
    signal_out = np.zeros(M * len(signal))  # output length
    phase_pre = np.ones(N)
    last_phase = np.ones(N)
    current_phase = np.zeros(N)
    b_k = []
    # %%

    for k in range(0, signal.shape[0] + 1 - N, step):
        # Analysis

        x = np.multiply(signal[k:k + N], np.hamming(N))
        a_k = FourierCoeffGen(x)
        # TODO: 1. Extract the Frame's phase.
        #       2. Find the diff phase
        phase = []
        phase_diff = []



        ############# Your code here ############
        ## (~2 line of code)
        abs_a_k = [0 for l in range(len(a_k))]
        for p in range(len(a_k)):
             abs_a_k[p] = (abs(a_k[p]))
        phase = np.divide(a_k, abs_a_k)
        phase_diff = np.divide(phase, phase_pre)


        #########################################

        for n in range(M):
            # Synthesis
            # TODO: 1. Compute the current signal's phase.
            #       2. Compute the output b_k
            #       3. Save the last phase for the next frame

            ############# Your code here ############
            ## (~3 line of code)

            current_phase = np.multiply(last_phase, phase_diff)
            b_k = np.multiply(abs_a_k, current_phase)
            last_phase = current_phase

            #########################################

            w = np.real(DiscreteFourierSeries(b_k))
            z = np.multiply(w, np.hamming(N))
            signal_out[kk:kk + N] = signal_out[kk:kk + N] + z
            kk = kk + step


        phase_pre = phase

    # %% cheack your results
    plt.figure(2)
    plt.title("Output signal Wave")
    plt.plot(signal_out)

    output_path = "C:/Users/ahiga/Desktop/python Signals&Systems/fire_out_M=2.wav"  # write your path here!
    sf.write(output_path, signal_out, fs)
    sd.play(signal_out, fs)

def partA ():

    x1 = []
    c1 = 5
    for n in range(c1):
        x1.append(np.cos(2*np.pi*n/c1))
    plt.stem(x1)
    plt.show()
    print(x1)
    x1_ak = FourierCoeffGen(x1)
    print(FourierCoeffGen(x1))
    plt.stem(np.abs(x1_ak))
    plt.show()


    x2 = []
    N1 = 2
    N = 20*N1

    for i in range(N):
        x2.append(0)
    for n in range(int(-N/2), int(N/2)):
        if ((abs(n)) < (5*N1)):
            x2[n+int(N/2)] = 1
    print(x2)

    x2_ak = FourierCoeffGen(x2)
    print(FourierCoeffGen(x2))
    plt.stem(x2)
    plt.show()
    plt.stem(x2_ak)
    plt.show()
    x1_xn = DiscreteFourierSeries(x1_ak)
    plt.stem(x1_xn)
    plt.show()
    x2_xn = DiscreteFourierSeries(x2_ak)
    plt.stem(x2_xn)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #partA()
    func()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

