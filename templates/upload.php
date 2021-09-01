<html>
<body>


<?php
if (isset($_POST['submit'])) {
  $file = $_FILES['file'];

  $fileName =$_FILES['file']['name'];
  $fileTmpName =$_FILES['file']['tmp_name'];
  $fileSize =$_FILES['file']['size'];
  $fileError =$_FILES['file']['error'];
  $fileType =$_FILES['file']['type'];

  $fileExt = explode('.', $fileName);
  $fileActualExt = strtolower(end($fileExt));

  $allowed = array('jpg', 'jpeg', 'png');

  if (in_array($fileActualExt, $allowed)) {
    if ($fileError === 0) {
      if ($fileSize < 500) {
        $fileNameNew = uniqid('',true).".".$fileActualExt;
        $fileDestination = 'uploads/'.$fileNameNew;
        move_uploaded_file($fileTmpName, $fileDestination );
        header("location: index.php?uploaded_success");
      }else{
        echo "your cannot over 500KB";
      }
    } else{
      echo "there was an error of your file";
    }
  } else{
    echo "wrong type of image";
  }
}
?>

</html>
</body>
