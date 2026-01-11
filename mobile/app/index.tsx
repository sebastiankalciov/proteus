import { useState } from "react";
import { Alert, Text, View, ActivityIndicator, Image, Button} from "react-native";

import * as ImagePicker from 'expo-image-picker';
const LOCAL_IP_ADDRESS = "http://192.168.1.162:8001";

export default function Index() {
  const [imageUri, setImageUri] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);

  const [permission, getPermission] = ImagePicker.useCameraPermissions();

  const takePicture = async () => {
    if (!permission?.granted) {
      const {granted} = await getPermission();
      if (!granted) {
        Alert.alert("permission needed", "da ba permisiune borfasule");
        return;
      }
    }

    const uploadImage = async (uri: string) => {
      setUploading(true);
      setPrediction(null);
      try {
        const formData = new FormData();
        const fileName = uri.split('/').pop();
        const fileType = fileName.split('.').pop();

        formData.append('file', {
          uri: uri,
          name: `photo.${fileType}`,
          type: `image/${fileType === 'jpg' ? 'jpeg' : fileType}`,
        });

        const response = await fetch(LOCAL_IP_ADDRESS + '/predict', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        
        if (response.ok) {
          setPrediction(data);
        } else {
          Alert.alert("Error", data.detail || "Server error");
        }
      } catch (error: any) {
        console.log("error: ", error.message);
        Alert.alert("Upload failed", error.message);
      } finally {
        setUploading(false);
      }
    }

    try {
      const imageResult = await ImagePicker.launchCameraAsync({
        allowsEditing: false,
        aspect: [4, 3],
        base64: false,
      });
      if (!imageResult.canceled) {
        setImageUri(imageResult.assets[0].uri);
        uploadImage(imageResult.assets[0].uri);
      }
    } catch (error: any) {
      Alert.alert("error", "eroare: " + error.message);
    }

  }



  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 }}>
      <Text style={{ fontSize: 24, marginBottom: 20 }}>Rock Paper Scissors</Text>

      {imageUri && (
        <Image
          source={{ uri: imageUri }}
          style={{ width: 280, height: 280, borderRadius: 12, marginBottom: 20 }}
        />
      )}

      {uploading ? (
        <ActivityIndicator size="large" color="#0000ff" />
      ) : (
        <Button title="take photo & predict" onPress={takePicture} />
      )}
      
      {prediction && (
        <View style={{ marginTop: 30, alignItems: 'center' }}>
          <Text style={{ fontSize: 28, fontWeight: 'bold' }}>
            {prediction.prediction}
          </Text>
          <Text style={{ fontSize: 18, marginTop: 8 }}>
            Confidence: {(prediction.confidence).toFixed(1)}%
          </Text>
        </View>
      )}
  </View>
  );
}
