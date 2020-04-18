import os
isim="sigara ("
sayi=0
while sayi <=285:
	if os.path.exists(isim+str(sayi)+").xml"):
		if not os.path.exists(isim+str(sayi)+").png"):
			os.remove(isim+str(sayi)+").xml")
			print(isim+str(sayi)+".xml"+" silindi.")
		else: print((isim+str(sayi)+").png dosyasi kontrol edildi."))
	sayi = sayi+1

